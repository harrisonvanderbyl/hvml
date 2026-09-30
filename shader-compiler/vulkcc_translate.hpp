// vulkcc_translate.hpp — CUDA-style device code → GLSL compute shaders.
//
// vulkcc parses the program as CUDA (clang, host side) and, for every kernel
// the program launches through vulkcc::launch<kernel>(...), translates the
// __global__ function and everything it calls from the instantiated AST to a
// GLSL compute shader:
//
//   * C++ structs become GLSL structs with the same byte layout (scalar block
//     layout; checked against clang's record layout).  Methods, constructors
//     and operators become free functions.
//   * Pointers are 64-bit device addresses (GL_EXT_buffer_reference): *p,
//     p[i], p->f and pointer arithmetic keep their C++ meaning.
//   * References to memory are passed as addresses, references to local
//     values as `inout` parameters, const references by value.  Functions are
//     generated once per combination of these.
//   * Kernel parameters live in an argument buffer whose address is the push
//     constant; they behave as memory.
//   * threadIdx / blockIdx / blockDim / gridDim, __shared__, __syncthreads,
//     atomics, warp shuffles (subgroup ops, with a shared-memory fallback
//     when the subgroup is narrower than the shuffle) and the math library map
//     to their GLSL / Vulkan equivalents.
//
// Anything outside that (unions, virtual calls, recursion, function pointers,
// pointers to locals or shared memory) makes the kernel "unsupported"; vulkcc
// skips it with a message and launching it reports that it was not compiled.

#pragma once

#include <clang/AST/AST.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Mangle.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vulkcc {

using namespace clang;

struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline std::string sanitize(const std::string& s) {
    std::string out;
    for (char c : s) {
        char k = isalnum((unsigned char)c) ? c : '_';
        if (k == '_' && !out.empty() && out.back() == '_') continue;   // GLSL reserves "__"
        out += k;
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (out.empty()) out = "x";
    return out;
}

// An lvalue in the generated code.
struct LV {
    enum Kind { Reg, Mem, Shared } kind = Reg;
    std::string text;      // Reg/Shared: GLSL lvalue;  Mem: uint64_t address expression
    QualType type;         // C++ type of the object
    bool u8bool = false;   // a bool stored as uint8_t (struct fields)
    bool readonly = false; // a copy standing in for a reinterpreted local (writes would be lost)
};

class Kernel {
public:
    Kernel(ASTContext& ctx, const SourceManager& sm) : Ctx(ctx), SM(sm) {}

    // GLSL source of `kernel` (a __global__ function instantiation).
    std::string translate(const FunctionDecl* kernel) {
        std::string body = entry(kernel);
        std::ostringstream s;
        s << "#version 460\n"
             "#extension GL_EXT_buffer_reference : require\n"
             "#extension GL_EXT_buffer_reference2 : require\n"
             "#extension GL_EXT_scalar_block_layout : require\n"
             "#extension GL_EXT_shader_explicit_arithmetic_types : require\n";
        if (use_subgroup) {
            s << "#extension GL_KHR_shader_subgroup_basic : require\n"
                 "#extension GL_KHR_shader_subgroup_shuffle : require\n"
                 "#extension GL_KHR_shader_subgroup_shuffle_relative : require\n"
                 "#extension GL_KHR_shader_subgroup_vote : require\n"
                 "#extension GL_KHR_shader_subgroup_ballot : require\n";
        }
        if (use_atomic_float) s << "#extension GL_EXT_shader_atomic_float : require\n";
        if (use_atomic_int64) s << "#extension GL_EXT_shader_atomic_int64 : require\n";
        s << "layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n"
             "layout(push_constant) uniform VkArgs { uint64_t vk_args; };\n\n";
        s << struct_decls << "\n" << block_decls << "\n" << shared_decls << "\n";
        s << helper_decls << "\n" << function_defs << "\n" << body;
        return s.str();
    }

private:
    ASTContext& Ctx;
    const SourceManager& SM;

    std::string struct_decls, block_decls, shared_decls, helper_decls, function_defs;
    std::map<const RecordDecl*, std::string> struct_names;
    std::set<const RecordDecl*> struct_in_progress;
    std::map<std::string, std::string> blocks;          // "type|align" → block name
    std::set<std::string> helpers;
    std::map<std::pair<const FunctionDecl*, std::string>, std::string> functions;
    std::set<const FunctionDecl*> function_in_progress;
    std::map<const VarDecl*, std::string> shared_names;
    bool use_subgroup = false, use_atomic_float = false, use_atomic_int64 = false, use_shuffle_scratch = false;
    int counter = 0;

    // ---- per-function state -------------------------------------------------
    struct Fn {
        std::map<const ValueDecl*, LV> vars;
        LV self;
        bool has_self = false;
        bool returns_address = false;
        bool returns_self = false;             // every return is *this (see returns_this)
        QualType return_type;
        // tail self-calls become a loop over the body (GLSL has no recursion)
        const FunctionDecl* decl = nullptr;
        std::string binding;
        std::vector<std::string> slots;        // parameter declarations, in binding order
        bool tail_loop = false;
    };
    Fn* fn = nullptr;

    // Lambdas: a local closure variable is not materialised; its captures
    // are bound when it is made (by-copy ones snapshotted into locals) and
    // passed to its call operator's function ahead of the arguments.
    struct Lambda {
        const LambdaExpr* expr = nullptr;
        const Fn* made_in = nullptr;          // captures are that function's locals
        std::vector<LV> captures;             // one per le->captures(), in order
    };
    std::map<const CXXRecordDecl*, Lambda> lambdas;
    const Lambda* lambda_for_function = nullptr;   // set while function() builds a call operator

    std::string fresh(const std::string& base) { return "v" + std::to_string(counter++) + "_" + sanitize(base); }

    // ======================================================================
    //  Types
    // ======================================================================

    static QualType bare(QualType t) { return t.getNonReferenceType().getCanonicalType().getUnqualifiedType(); }

    long size_of(QualType t) { return (long)Ctx.getTypeSizeInChars(bare(t)).getQuantity(); }
    long align_of(QualType t) { return (long)Ctx.getTypeAlignInChars(bare(t)).getQuantity(); }

    // GLSL type of a C++ type (arrays are handled by decl()).
    std::string type(QualType t) {
        t = bare(t);
        if (t->isVoidType()) return "void";
        if (t->isPointerType() || t->isNullPtrType()) return "uint64_t";
        if (const auto* et = t->getAs<EnumType>()) return type(et->getDecl()->getIntegerType());
        if (const auto* bt = t->getAs<BuiltinType>()) {
            switch (bt->getKind()) {
                case BuiltinType::Bool: return "bool";
                case BuiltinType::Char_S: case BuiltinType::SChar: return "int8_t";
                case BuiltinType::Char_U: case BuiltinType::UChar: return "uint8_t";
                case BuiltinType::Short: return "int16_t";
                case BuiltinType::UShort: return "uint16_t";
                case BuiltinType::Int: return "int";
                case BuiltinType::UInt: return "uint";
                case BuiltinType::Long: case BuiltinType::LongLong: return "int64_t";
                case BuiltinType::ULong: case BuiltinType::ULongLong: return "uint64_t";
                case BuiltinType::Float: return "float";
                case BuiltinType::Double: return "double";
                case BuiltinType::Half: case BuiltinType::Float16: return "float16_t";
                default: break;
            }
        }
        if (const auto* rt = t->getAs<RecordType>()) return record(rt->getDecl());
        if (t->isArrayType()) throw Unsupported("array type in an expression context: " + t.getAsString());
        throw Unsupported("type " + t.getAsString() + " has no GLSL equivalent");
    }

    // "T name[N][M]"
    std::string decl(QualType t, const std::string& name, bool in_memory = false) {
        t = bare(t);
        std::string suffix;
        while (const auto* at = Ctx.getAsConstantArrayType(t)) {
            suffix += "[" + std::to_string(at->getSize().getZExtValue()) + "]";
            t = bare(at->getElementType());
        }
        if (t->isArrayType()) throw Unsupported("variable-length array");
        std::string base = (in_memory && t->isBooleanType()) ? "uint8_t" : type(t);
        return base + " " + name + suffix;
    }

    static std::string zero_scalar(const std::string& t) {
        if (t == "bool") return "false";
        if (t == "float") return "0.0";
        if (t == "double") return "0.0lf";
        if (t == "int") return "0";
        if (t == "uint") return "0u";
        if (t == "int64_t") return "0l";
        if (t == "uint64_t") return "0ul";
        return t + "(0)";
    }

    std::string zero(QualType t) {
        t = bare(t);
        if (const auto* at = Ctx.getAsConstantArrayType(t)) {
            if (Ctx.getAsConstantArrayType(bare(at->getElementType()))) throw Unsupported("nested array initialiser");
            std::string e = zero(at->getElementType());
            uint64_t n = at->getSize().getZExtValue();
            std::string list;
            for (uint64_t i = 0; i < n; i++) list += (i ? ", " : "") + e;
            return type(bare(at->getElementType())) + "[" + std::to_string(n) + "](" + list + ")";
        }
        if (const auto* rt = t->getAs<RecordType>()) {
            const RecordDecl* rd = rt->getDecl()->getDefinition();
            std::string name = record(rd);
            std::vector<std::string> parts;
            for (auto& f : record_fields(rd)) parts.push_back(f.in_memory_bool ? "uint8_t(0)" : zero(f.type));
            std::string out = name + "(";
            for (size_t i = 0; i < parts.size(); i++) out += (i ? ", " : "") + parts[i];
            return out + ")";
        }
        return zero_scalar(type(t));
    }

    // Fields of the GLSL struct for a record (bases first, as `base<i>`).
    struct FieldInfo {
        std::string name;
        QualType type;
        const FieldDecl* field = nullptr;       // null for a base
        const CXXRecordDecl* base = nullptr;
        long offset = 0;
        bool in_memory_bool = false;
    };
    std::map<const RecordDecl*, std::vector<FieldInfo>> field_cache;

    const std::vector<FieldInfo>& record_fields(const RecordDecl* rd) {
        rd = rd->getDefinition();
        auto found = field_cache.find(rd);
        if (found != field_cache.end()) return found->second;
        std::vector<FieldInfo> out;
        const ASTRecordLayout& layout = Ctx.getASTRecordLayout(rd);
        if (rd->isUnion()) throw Unsupported("union " + rd->getNameAsString());
        if (const auto* cxx = dyn_cast<CXXRecordDecl>(rd)) {
            if (cxx->isPolymorphic()) throw Unsupported("class with virtual functions: " + cxx->getNameAsString());
            int i = 0;
            for (const CXXBaseSpecifier& b : cxx->bases()) {
                const CXXRecordDecl* bd = b.getType()->getAsCXXRecordDecl();
                if (b.isVirtual()) throw Unsupported("virtual base");
                if (!bd || bd->isEmpty()) { i++; continue; }
                FieldInfo f;
                f.name = "base" + std::to_string(i++);
                f.type = b.getType();
                f.base = bd;
                f.offset = (long)layout.getBaseClassOffset(bd).getQuantity();
                out.push_back(f);
            }
        }
        for (const FieldDecl* fd : rd->fields()) {
            if (fd->isBitField()) throw Unsupported("bit-field " + fd->getNameAsString());
            FieldInfo f;
            f.name = "m_" + sanitize(fd->getNameAsString());
            f.type = fd->getType();
            f.field = fd;
            f.offset = (long)(layout.getFieldOffset(fd->getFieldIndex()) / 8);
            QualType ft = bare(fd->getType());
            while (const auto* at = Ctx.getAsConstantArrayType(ft)) ft = bare(at->getElementType());
            f.in_memory_bool = ft->isBooleanType();
            if (fd->getType()->isReferenceType()) throw Unsupported("reference member " + fd->getNameAsString());
            out.push_back(f);
        }
        return field_cache[rd] = out;
    }

    // scalar-layout alignment and size of a GLSL-mapped type
    long glsl_align(QualType t) {
        t = bare(t);
        if (const auto* at = Ctx.getAsConstantArrayType(t)) return glsl_align(at->getElementType());
        if (const auto* rt = t->getAs<RecordType>()) {
            long a = 1;
            for (auto& f : record_fields(rt->getDecl())) a = std::max(a, glsl_align(f.type));
            return a;
        }
        return size_of(t);
    }

    std::string record(const RecordDecl* rd) {
        rd = rd->getDefinition();
        if (!rd) throw Unsupported("incomplete type");
        auto found = struct_names.find(rd);
        if (found != struct_names.end()) return found->second;
        if (struct_in_progress.count(rd)) throw Unsupported("recursive struct " + rd->getNameAsString());
        struct_in_progress.insert(rd);

        std::string name = "S" + std::to_string(counter++) + "_" + sanitize(rd->getNameAsString());
        const auto& fields = record_fields(rd);
        std::string def = "struct " + name + " {\n";
        long off = 0, align = 1;
        for (auto& f : fields) {
            std::string d = decl(f.type, f.name, true);   // resolves nested structs first
            long a = glsl_align(f.type);
            off = (off + a - 1) / a * a;
            if (off != f.offset) {
                throw Unsupported("layout of " + rd->getNameAsString() + " (member " + f.name +
                                  " at C++ offset " + std::to_string(f.offset) + ", GLSL " + std::to_string(off) + ")");
            }
            off += size_of(f.type);
            align = std::max(align, a);
            def += "    " + d + ";\n";
        }
        if (fields.empty()) {
            def += "    uint8_t m_empty;\n";
            off = 1;
        }
        long size = (off + align - 1) / align * align;
        long cxx_size = (long)Ctx.getASTRecordLayout(rd).getSize().getQuantity();
        if (size != cxx_size) {
            throw Unsupported("size of " + rd->getNameAsString() + " (C++ " + std::to_string(cxx_size) + ", GLSL " +
                              std::to_string(size) + ")");
        }
        def += "};\n";
        struct_decls += def;
        struct_in_progress.erase(rd);
        return struct_names[rd] = name;
    }

    // buffer_reference block for loads/stores of `t` at an address
    std::string block(QualType t) {
        t = bare(t);
        std::string elem = t->isBooleanType() ? "uint8_t" : type(t);
        long align = std::max(1L, align_of(t));
        std::string key = elem + "|" + std::to_string(align);
        auto found = blocks.find(key);
        if (found != blocks.end()) return found->second;
        std::string name = "P" + std::to_string(blocks.size()) + "_" + sanitize(elem);
        block_decls += "layout(buffer_reference, scalar, buffer_reference_align = " + std::to_string(align) +
                       ") buffer " + name + " { " + elem + " v[]; };\n";
        return blocks[key] = name;
    }

    // ======================================================================
    //  Lvalues
    // ======================================================================

    std::string mem_ref(const LV& lv) { return block(lv.type) + "(" + lv.text + ").v[0]"; }

    std::string load(const LV& lv) {
        std::string ref = lv.kind == LV::Mem ? mem_ref(lv) : lv.text;
        if ((lv.kind == LV::Mem && bare(lv.type)->isBooleanType()) || lv.u8bool) return "(" + ref + " != uint8_t(0))";
        return ref;
    }

    std::string writable_place(const LV& lv) {
        if (lv.readonly) throw Unsupported("passing a reinterpreted local value by non-const reference");
        return place(lv);
    }

    // GLSL lvalue text to assign / pass as inout
    std::string place(const LV& lv) { return lv.kind == LV::Mem ? mem_ref(lv) : lv.text; }

    std::string store(const LV& lv, const std::string& value) {
        if (lv.readonly) throw Unsupported("writing through a reinterpreted local value (e.g. local.xyz() = ...); write the elements");
        bool as_u8 = (lv.kind == LV::Mem && bare(lv.type)->isBooleanType()) || lv.u8bool;
        if (as_u8) return place(lv) + " = ((" + value + ") ? uint8_t(1) : uint8_t(0))";
        return place(lv) + " = " + value;
    }

    std::string address(const LV& lv, const char* what) {
        if (lv.kind != LV::Mem) {
            throw Unsupported(std::string("taking the address of a local or shared variable (") + what + ")");
        }
        return lv.text;
    }

    LV field_of(const LV& base, const std::string& glsl_field, long offset, QualType ftype, bool u8bool) {
        LV r;
        r.type = ftype;
        r.kind = base.kind;
        r.readonly = base.readonly;
        if (base.kind == LV::Mem) {
            r.text = offset ? "(" + base.text + " + " + std::to_string(offset) + "ul)" : base.text;
        } else {
            r.text = base.text + "." + glsl_field;
            r.u8bool = u8bool;
        }
        return r;
    }

    LV base_of(const LV& obj, const CXXRecordDecl* derived, const CXXRecordDecl* base_rd, QualType base_type) {
        for (auto& f : record_fields(derived)) {
            if (f.base && f.base->getCanonicalDecl() == base_rd->getCanonicalDecl()) {
                return field_of(obj, f.name, f.offset, base_type, false);
            }
        }
        // empty base or through several levels
        if (obj.kind == LV::Mem) {
            LV r = obj;
            r.type = base_type;
            r.text = "(" + obj.text + " + " +
                     std::to_string(Ctx.getASTRecordLayout(derived).getBaseClassOffset(base_rd).getQuantity()) + "ul)";
            return r;
        }
        throw Unsupported("indirect base class of a local value");
    }

    LV lvalue(const Expr* e) {
        e = e->IgnoreParens();
        if (const auto* x = dyn_cast<ExprWithCleanups>(e)) return lvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<MaterializeTemporaryExpr>(e)) {
            // a temporary bound to a reference: give it a local
            return temporary(x->getSubExpr());
        }
        if (const auto* x = dyn_cast<CXXBindTemporaryExpr>(e)) return lvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<ConstantExpr>(e)) return lvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<SubstNonTypeTemplateParmExpr>(e)) return lvalue(x->getReplacement());

        if (const auto* dr = dyn_cast<DeclRefExpr>(e)) {
            const ValueDecl* d = dr->getDecl();
            auto found = fn->vars.find(d);
            if (found != fn->vars.end()) return found->second;
            if (const auto* vd = dyn_cast<VarDecl>(d)) {
                if (vd->hasAttr<CUDASharedAttr>()) {
                    LV r;
                    r.kind = LV::Shared;
                    r.text = shared(vd);
                    r.type = vd->getType();
                    return r;
                }
            }
            throw Unsupported("reference to " + d->getQualifiedNameAsString() + " (globals are not supported)");
        }
        if (const auto* me = dyn_cast<MemberExpr>(e)) {
            const ValueDecl* member = me->getMemberDecl();
            const auto* fd = dyn_cast<FieldDecl>(member);
            if (!fd) throw Unsupported("member " + member->getNameAsString());
            LV obj;
            if (me->isArrow()) {
                if (isa<CXXThisExpr>(me->getBase()->IgnoreParenImpCasts())) obj = self();
                else {
                    obj.kind = LV::Mem;
                    obj.text = rvalue(me->getBase());
                    obj.type = me->getBase()->getType()->getPointeeType();
                }
            } else {
                obj = lvalue_or_temp(me->getBase());
            }
            const RecordDecl* rd = fd->getParent();
            for (auto& f : record_fields(rd)) {
                if (f.field && f.field->getCanonicalDecl() == fd->getCanonicalDecl()) {
                    return field_of(obj, f.name, f.offset, fd->getType(), f.in_memory_bool);
                }
            }
            throw Unsupported("field " + fd->getNameAsString());
        }
        if (const auto* as = dyn_cast<ArraySubscriptExpr>(e)) {
            const Expr* base = as->getBase()->IgnoreParens();
            std::string idx = rvalue(as->getIdx());
            QualType elem = as->getType();
            if (const auto* ic = dyn_cast<ImplicitCastExpr>(base)) {
                if (ic->getCastKind() == CK_ArrayToPointerDecay) {
                    LV arr = lvalue_or_temp(ic->getSubExpr());
                    LV r;
                    r.kind = arr.kind;
                    r.readonly = arr.readonly;
                    r.type = elem;
                    if (arr.kind == LV::Mem) r.text = "(" + arr.text + " + uint64_t(int64_t(" + idx + ")) * " + std::to_string(size_of(elem)) + "ul)";
                    else {
                        // GLSL indexes arrays with 32-bit ints only (size_t loop
                        // counters, uint8_t swizzle components...).
                        QualType it = as->getIdx()->getType().getCanonicalType();
                        bool is_int32 = it->isIntegerType() && Ctx.getTypeSize(it) == 32;
                        r.text = arr.text + "[" + (is_int32 ? idx : "int(" + idx + ")") + "]";
                        r.u8bool = arr.u8bool;
                    }
                    return r;
                }
            }
            LV r;
            r.kind = LV::Mem;
            r.type = elem;
            r.text = "(" + rvalue(base) + " + uint64_t(int64_t(" + idx + ")) * " + std::to_string(size_of(elem)) + "ul)";
            return r;
        }
        if (const auto* uo = dyn_cast<UnaryOperator>(e)) {
            if (uo->getOpcode() == UO_Deref) {
                if (isa<CXXThisExpr>(uo->getSubExpr()->IgnoreParenImpCasts())) return self();
                std::string punned;
                if (pun(uo, punned)) {                 // *(T*)&local, read as T
                    LV t;
                    t.kind = LV::Reg;
                    t.text = fresh("pun");
                    t.type = e->getType();
                    t.readonly = true;
                    emit_pending(decl(e->getType(), t.text) + " = " + punned + ";");
                    return t;
                }
                LV r;
                r.kind = LV::Mem;
                r.text = rvalue(uo->getSubExpr());
                r.type = e->getType();
                return r;
            }
            if (uo->getOpcode() == UO_PreInc || uo->getOpcode() == UO_PreDec) {
                LV target = lvalue(uo->getSubExpr());
                emit_pending(rvalue(uo) + ";");
                return target;
            }
        }
        if (const auto* ic = dyn_cast<ImplicitCastExpr>(e)) {
            switch (ic->getCastKind()) {
                case CK_NoOp: case CK_LValueBitCast: return retype(lvalue(ic->getSubExpr()), e->getType());
                case CK_UncheckedDerivedToBase: case CK_DerivedToBase: {
                    LV obj = lvalue(ic->getSubExpr());
                    const CXXRecordDecl* derived = ic->getSubExpr()->getType()->getAsCXXRecordDecl();
                    return base_of(obj, derived, e->getType()->getAsCXXRecordDecl(), e->getType());
                }
                default: break;
            }
        }
        if (const auto* ce = dyn_cast<CastExpr>(e)) {
            if (ce->getCastKind() == CK_NoOp || ce->getCastKind() == CK_LValueBitCast)
                return retype(lvalue(ce->getSubExpr()), e->getType());
        }
        if (const auto* co = dyn_cast<ConditionalOperator>(e)) {
            LV a = lvalue(co->getTrueExpr()), b = lvalue(co->getFalseExpr());
            if (a.kind == LV::Mem && b.kind == LV::Mem) {
                LV r = a;
                r.text = "(" + rvalue(co->getCond()) + " ? " + a.text + " : " + b.text + ")";
                return r;
            }
            throw Unsupported("conditional lvalue of local values");
        }
        if (const auto* bo = dyn_cast<BinaryOperator>(e)) {
            if (bo->isAssignmentOp()) {
                LV target = lvalue(bo->getLHS());
                emit_pending(rvalue(bo) + ";");
                return target;
            }
        }
        if (const auto* ce = dyn_cast<CallExpr>(e)) {
            if (const Expr* target = trivial_assign_target(ce)) {
                emit_pending(call(ce) + ";");
                return lvalue(target);
            }
            if (const FunctionDecl* fd = ce->getDirectCallee()) {
                if (fd->isInStdNamespace() && (fd->getName() == "forward" || fd->getName() == "move"))
                    return retype(lvalue(ce->getArg(0)), e->getType());
            }
            QualType rt = ce->getCallReturnType(Ctx);
            if (rt->isLValueReferenceType() && !rt.getNonReferenceType().isConstQualified()) {
                LV inlined;
                if (inline_accessor(ce, inlined)) return inlined;
                if (const FunctionDecl* callee = ce->getDirectCallee(); callee && returns_this(callee)) {
                    unsigned first = 0;
                    const Expr* obj = call_object(ce, llvm::cast<CXXMethodDecl>(callee), first);
                    if (obj) {
                        emit_pending(call(ce) + ";");
                        obj = obj->IgnoreParens();
                        if (isa<CXXThisExpr>(obj->IgnoreParenImpCasts())) return retype(self(), e->getType());
                        if (obj->getType()->isPointerType()) {
                            LV r;
                            r.kind = LV::Mem;
                            r.text = rvalue(obj);
                            r.type = e->getType();
                            return r;
                        }
                        return retype(lvalue(obj), e->getType());
                    }
                }
                LV r;
                r.kind = LV::Mem;
                r.text = call(ce);
                r.type = e->getType();
                return r;
            }
            return temporary(e);
        }
        if (e->isPRValue()) return temporary(e);
        throw Unsupported(std::string("lvalue ") + e->getStmtClassName());
    }

    // A call to a function whose body is just `return <lvalue>;` — accessors
    // such as operator[] or a swizzle's x() returning T& — evaluated in the
    // caller: the object and reference arguments bind to their lvalues, value
    // arguments to temporaries, and the returned lvalue is built from those.
    // A GLSL function can only return an address, which a local object does
    // not have; in place, `local[i]` and `local.x()` work like memory ones.
    bool inline_accessor(const CallExpr* ce, LV& out) {
        const FunctionDecl* callee = ce->getDirectCallee();
        const FunctionDecl* def = nullptr;
        if (!callee || !callee->hasBody(def) || !def) return false;
        const auto* body = dyn_cast_or_null<CompoundStmt>(def->getBody());
        if (!body || body->size() != 1) return false;
        const auto* rs = dyn_cast<ReturnStmt>(body->body_front());
        if (!rs || !rs->getRetValue()) return false;

        const auto* md = dyn_cast<CXXMethodDecl>(def);
        bool member = md && md->isInstance();
        Fn inner;
        unsigned first = 0;
        if (member) {
            const Expr* obj = nullptr;
            if (const auto* mc = dyn_cast<CXXMemberCallExpr>(ce)) obj = mc->getImplicitObjectArgument();
            else if (isa<CXXOperatorCallExpr>(ce)) { obj = ce->getArg(0); first = 1; }
            if (!obj) return false;
            obj = obj->IgnoreParens();
            if (isa<CXXThisExpr>(obj->IgnoreParenImpCasts())) {
                inner.self = self();
            } else if (obj->getType()->isPointerType()) {        // p->f()
                inner.self.kind = LV::Mem;
                inner.self.text = rvalue(obj);
                inner.self.type = obj->getType()->getPointeeType();
            } else {
                inner.self = lvalue_or_temp(obj);
            }
            inner.has_self = true;
        }
        if (ce->getNumArgs() - first != def->getNumParams()) return false;
        for (unsigned i = 0; i < def->getNumParams(); i++) {
            const ParmVarDecl* p = def->getParamDecl(i);
            const Expr* arg = ce->getArg(first + i);
            if (p->getType()->isReferenceType()) {
                inner.vars[p] = lvalue_or_temp(arg);
            } else {
                LV t = temporary(arg);
                t.type = p->getType();
                inner.vars[p] = t;
            }
        }
        inner.return_type = def->getReturnType();

        Fn* caller = fn;
        fn = &inner;
        try {
            out = lvalue(rs->getRetValue());
        } catch (...) {
            fn = caller;
            throw;
        }
        fn = caller;
        out.type = ce->getType();
        return true;
    }

    // `dst = src` for C++ arrays (GLSL arrays are values: a local source is
    // assigned whole; one in memory is loaded element by element).
    std::string copy_array(const LV& dst, const LV& src, QualType array_type) {
        const auto* at = Ctx.getAsConstantArrayType(bare(array_type));
        if (!at) throw Unsupported("array copy of a non-array");
        if (dst.kind != LV::Mem && src.kind != LV::Mem) return place(dst) + " = " + place(src);
        QualType elem = at->getElementType();
        uint64_t n = at->getSize().getZExtValue();
        std::string k = fresh("k");
        auto element = [&](const LV& a) {
            LV e;
            e.kind = a.kind;
            e.type = elem;
            if (a.kind == LV::Mem) e.text = "(" + a.text + " + uint64_t(" + k + ") * " + std::to_string(size_of(elem)) + "ul)";
            else { e.text = a.text + "[" + k + "]"; e.u8bool = a.u8bool; }
            return e;
        };
        LV de = element(dst), se = element(src);
        std::string inner = Ctx.getAsConstantArrayType(bare(elem)) ? copy_array(de, se, elem) : store(de, load(se));
        return "for (int " + k + " = 0; " + k + " < " + std::to_string(n) + "; " + k + "++) { " + inner + "; }";
    }

    LV retype(LV lv, QualType t) {
        if (lv.kind == LV::Mem || bare(lv.type) == bare(t)) {
            lv.type = t;
            return lv;
        }
        throw Unsupported("reinterpreting a local value");
    }

    LV lvalue_or_temp(const Expr* e) {
        if (e->isPRValue()) return temporary(e);
        return lvalue(e);
    }

    // A prvalue that needs to be an lvalue (member access on a temporary,
    // a temporary bound to a reference): store it in a fresh local.
    LV temporary(const Expr* e) {
        std::string name = fresh("tmp");
        emit_pending(decl(e->getType(), name) + " = " + rvalue(e) + ";");
        LV r;
        r.kind = LV::Reg;
        r.text = name;
        r.type = e->getType();
        return r;
    }

    LV self() {
        if (!fn->has_self) throw Unsupported("this outside a member function");
        return fn->self;
    }

    std::string shared(const VarDecl* vd) {
        auto found = shared_names.find(vd);
        if (found != shared_names.end()) return found->second;
        std::string name = "sh" + std::to_string(shared_names.size()) + "_" + sanitize(vd->getNameAsString());
        shared_decls += "shared " + decl(vd->getType(), name) + ";\n";
        return shared_names[vd] = name;
    }

    // Statements that must run before the expression being built (temporaries).
    std::vector<std::string> pending;
    void emit_pending(const std::string& s) { pending.push_back(s); }

    // ======================================================================
    //  Rvalues
    // ======================================================================

    static bool is_float_type(const std::string& t) { return t == "float" || t == "double" || t == "float16_t"; }

    std::string convert(const std::string& v, const std::string& from, const std::string& to) {
        if (from == to || to == "void") return v;
        if (to == "bool") return "(" + v + " != " + zero_scalar(from) + ")";
        return to + "(" + v + ")";
    }

    std::string rvalue_as(const Expr* e, QualType to) {
        std::string from = type(e->getType());
        std::string t = type(to);
        return convert(rvalue(e), from, t);
    }

    std::string literal_int(const llvm::APSInt& v, QualType t) {
        std::string ty = type(t);
        std::string s = llvm::toString(v, 10);
        if (ty == "uint") return s + "u";
        if (ty == "int64_t") return s + "l";
        if (ty == "uint64_t") return s + "ul";
        if (ty == "int") {
            if (v.isSigned() && v.getSExtValue() == INT32_MIN) return "int(0x80000000u)";
            return s;
        }
        if (ty == "bool") return v.getBoolValue() ? "true" : "false";
        return ty + "(" + s + ")";
    }

    std::string literal_float(double v, QualType t) {
        std::string ty = type(t);
        if (std::isinf(v)) {
            std::string inf = v > 0 ? "uintBitsToFloat(0x7f800000u)" : "uintBitsToFloat(0xff800000u)";
            return ty == "float" ? inf : ty + "(" + inf + ")";
        }
        if (std::isnan(v)) return ty == "float" ? "uintBitsToFloat(0x7fc00000u)" : ty + "(uintBitsToFloat(0x7fc00000u))";
        char buf[64];
        snprintf(buf, sizeof buf, "%.17g", v);
        std::string s = buf;
        if (s.find_first_of(".e") == std::string::npos) s += ".0";
        if (ty == "double") return s + "lf";
        if (ty == "float") {
            snprintf(buf, sizeof buf, "%.9g", v);
            s = buf;
            if (s.find_first_of(".e") == std::string::npos) s += ".0";
            return s;
        }
        return ty + "(" + s + ")";
    }

    bool constant(const Expr* e, std::string& out) {
        if (e->isValueDependent() || e->isTypeDependent()) return false;
        Expr::EvalResult r;
        if (!e->EvaluateAsRValue(r, Ctx) || r.HasSideEffects) return false;
        if (r.Val.isInt()) { out = literal_int(r.Val.getInt(), e->getType()); return true; }
        if (r.Val.isFloat()) { out = literal_float(r.Val.getFloat().convertToDouble(), e->getType()); return true; }
        return false;
    }

    std::string cast(CastKind kind, const Expr* sub, QualType to) {
        std::string t = type(to);
        switch (kind) {
            case CK_LValueToRValue: {
                std::string punned;
                if (const auto* uo = dyn_cast<UnaryOperator>(sub->IgnoreParens()); uo && pun(uo, punned)) return punned;
                if (const auto* co = dyn_cast<ConditionalOperator>(sub->IgnoreParens())) {
                    // the value of an lvalue conditional
                    std::string t = type(to);
                    auto side = [&](const Expr* x) { return x->isGLValue() ? load(lvalue(x)) : rvalue(x); };
                    return "(" + rvalue_as(co->getCond(), Ctx.BoolTy) + " ? " + side(co->getTrueExpr()) + " : " +
                           side(co->getFalseExpr()) + ")";
                }
                return load(lvalue(sub));
            }
            case CK_NoOp: case CK_UserDefinedConversion: case CK_ConstructorConversion:
                return sub->isGLValue() ? load(lvalue(sub)) : rvalue(sub);
            case CK_ToVoid: return "";
            case CK_NullToPointer: case CK_NullToMemberPointer: return "0ul";
            case CK_BitCast: case CK_AddressSpaceConversion: return rvalue(sub);   // pointer → pointer
            case CK_ArrayToPointerDecay: return address(lvalue(sub), "array decay");
            case CK_PointerToBoolean: return "(" + rvalue(sub) + " != 0ul)";
            case CK_PointerToIntegral: case CK_IntegralToPointer:
                return convert(rvalue(sub), type(sub->getType()), t);
            case CK_DerivedToBase: case CK_UncheckedDerivedToBase: {
                if (to->isPointerType()) {
                    const CXXRecordDecl* d = sub->getType()->getPointeeCXXRecordDecl();
                    const CXXRecordDecl* b = to->getPointeeCXXRecordDecl();
                    return "(" + rvalue(sub) + " + " +
                           std::to_string(Ctx.getASTRecordLayout(d).getBaseClassOffset(b).getQuantity()) + "ul)";
                }
                LV obj = lvalue_or_temp(sub);
                return load(base_of(obj, sub->getType()->getAsCXXRecordDecl(), to->getAsCXXRecordDecl(), to));
            }
            case CK_LValueBitCast: case CK_LValueToRValueBitCast: {
                // reinterpret_cast<T&>(x): a bit cast of the value
                return bitcast(load(lvalue(sub)), type(sub->getType()), t);
            }
            case CK_IntegralCast: case CK_IntegralToFloating: case CK_FloatingToIntegral: case CK_FloatingCast:
            case CK_IntegralToBoolean: case CK_FloatingToBoolean: case CK_BooleanToSignedIntegral:
            case CK_FloatingComplexToReal: case CK_IntegralComplexToReal:
                return convert(rvalue(sub), type(sub->getType()), t);
            default:
                throw Unsupported(std::string("cast ") + CastExpr::getCastKindName(kind));
        }
    }

    std::string bitcast(const std::string& v, const std::string& from, const std::string& to) {
        if (from == to) return v;
        if (from == "float" && to == "uint") return "floatBitsToUint(" + v + ")";
        if (from == "float" && to == "int") return "floatBitsToInt(" + v + ")";
        if (from == "uint" && to == "float") return "uintBitsToFloat(" + v + ")";
        if (from == "int" && to == "float") return "intBitsToFloat(" + v + ")";
        if (from == "int" && to == "uint") return "uint(" + v + ")";
        if (from == "uint" && to == "int") return "int(" + v + ")";
        if (from == "double" && to == "uint64_t") return "doubleBitsToUint64(" + v + ")";
        if (from == "double" && to == "int64_t") return "doubleBitsToInt64(" + v + ")";
        if (from == "uint64_t" && to == "double") return "uint64BitsToDouble(" + v + ")";
        if (from == "int64_t" && to == "double") return "int64BitsToDouble(" + v + ")";
        if ((from == "uint64_t" && to == "int64_t") || (from == "int64_t" && to == "uint64_t")) return to + "(" + v + ")";
        throw Unsupported("bit cast from " + from + " to " + to);
    }

    // *(T*)&x  → bit cast of x
    bool pun(const UnaryOperator* uo, std::string& out) {
        if (uo->getOpcode() != UO_Deref) return false;
        const Expr* p = uo->getSubExpr()->IgnoreParens();
        while (const auto* ce = dyn_cast<CastExpr>(p)) {
            if (ce->getCastKind() != CK_BitCast && ce->getCastKind() != CK_NoOp) break;
            p = ce->getSubExpr()->IgnoreParens();
        }
        LV src;
        QualType from_t;
        if (isa<CXXThisExpr>(p)) {                       // *(U*)this
            if (!fn->has_self) return false;
            src = fn->self;
            from_t = bare(src.type);
        } else {
            const auto* addr = dyn_cast<UnaryOperator>(p);
            if (!addr || addr->getOpcode() != UO_AddrOf) return false;
            src = lvalue(addr->getSubExpr());
            from_t = bare(addr->getSubExpr()->getType());
        }
        if (src.kind == LV::Mem) return false;   // real memory: a typed load works
        QualType to_t = bare(uo->getType());
        if (from_t == to_t) { out = load(src); return true; }
        if (is_scalar_value(from_t) && is_scalar_value(to_t)) {
            out = bitcast(load(src), type(from_t), type(to_t));
            return true;
        }
        // a view of the leading part (Hvec swizzles: *(Hvec<T, 3, s>*)this on
        // an Hvec<T, 4>): built from the source's fields at the same offsets
        {
            std::string sv = fresh("src");
            std::vector<Leaf> leaves;
            flatten(from_t, sv, 0, src.u8bool, leaves);
            std::string built;
            if (build_from_leaves(to_t, 0, false, leaves, built)) {
                emit_pending(decl(from_t, sv) + " = " + load(src) + ";");
                out = built;
                return true;
            }
        }
        // structs / arrays of up to 64 bits: pack the source into a uint64_t
        // at its fields' byte offsets, unpack the target from it
        long n = size_of(from_t);
        if (n != size_of(to_t) || n > 8) throw Unsupported("reinterpreting a local value of " + std::to_string(n) + " bytes");
        std::string bits = fresh("bits");
        emit_pending("uint64_t " + bits + " = " + pack_bits(load(src), from_t, 0) + ";");
        out = unpack_bits(bits, to_t, 0, false);
        return true;
    }

    struct Leaf {
        long offset;
        QualType type;
        std::string expr;
        bool u8;
    };

    void flatten(QualType t, const std::string& e, long offset, bool u8, std::vector<Leaf>& out) {
        t = bare(t);
        if (const auto* at = Ctx.getAsConstantArrayType(t)) {
            long es = size_of(at->getElementType());
            for (uint64_t i = 0; i < at->getSize().getZExtValue(); i++)
                flatten(at->getElementType(), e + "[" + std::to_string(i) + "]", offset + (long)i * es, false, out);
            return;
        }
        if (const auto* rt = t->getAs<RecordType>()) {
            for (auto& f : record_fields(rt->getDecl()->getDefinition()))
                flatten(f.type, e + "." + f.name, offset + f.offset, f.in_memory_bool, out);
            return;
        }
        out.push_back({offset, t, e, u8});
    }

    bool build_from_leaves(QualType t, long offset, bool u8, const std::vector<Leaf>& leaves, std::string& out) {
        t = bare(t);
        if (const auto* at = Ctx.getAsConstantArrayType(t)) {
            std::string r = type(bare(at->getElementType())) + "[" + std::to_string(at->getSize().getZExtValue()) + "](";
            long es = size_of(at->getElementType());
            for (uint64_t i = 0; i < at->getSize().getZExtValue(); i++) {
                std::string v;
                if (!build_from_leaves(at->getElementType(), offset + (long)i * es, false, leaves, v)) return false;
                r += (i ? ", " : "") + v;
            }
            out = r + ")";
            return true;
        }
        if (const auto* rt = t->getAs<RecordType>()) {
            const RecordDecl* rd = rt->getDecl()->getDefinition();
            const auto& fields = record_fields(rd);
            std::string r = record(rd) + "(";
            for (size_t i = 0; i < fields.size(); i++) {
                std::string v;
                if (!build_from_leaves(fields[i].type, offset + fields[i].offset, fields[i].in_memory_bool, leaves, v)) return false;
                r += (i ? ", " : "") + v;
            }
            if (fields.empty()) r += "uint8_t(0)";
            out = r + ")";
            return true;
        }
        for (const Leaf& l : leaves) {
            if (l.offset == offset && bare(l.type) == t && l.u8 == u8) { out = l.expr; return true; }
        }
        return false;
    }

    bool is_scalar_value(QualType t) {
        t = bare(t);
        return t->isScalarType() && !t->isMemberPointerType();
    }

    // Bits of a scalar as uint64_t (low `size` bytes).
    std::string scalar_bits(const std::string& v, QualType t, bool as_u8) {
        t = bare(t);
        if (as_u8) return "uint64_t(uint8_t(" + v + "))";
        std::string g = type(t);
        if (g == "bool") return "uint64_t((" + v + ") ? 1u : 0u)";
        if (g == "float") return "uint64_t(floatBitsToUint(" + v + "))";
        if (g == "float16_t") return "uint64_t(float16BitsToUint16(" + v + "))";
        if (g == "double") return "doubleBitsToUint64(" + v + ")";
        if (g == "int8_t") return "uint64_t(uint8_t(" + v + "))";
        if (g == "int16_t") return "uint64_t(uint16_t(" + v + "))";
        if (g == "int") return "uint64_t(uint(" + v + "))";
        if (g == "int64_t") return "uint64_t(" + v + ")";
        return "uint64_t(" + v + ")";   // unsigned types, pointers
    }

    std::string scalar_from_bits(const std::string& bits, QualType t, bool as_u8) {
        t = bare(t);
        if (as_u8) return "uint8_t(" + bits + ")";
        std::string g = type(t);
        if (g == "bool") return "(uint8_t(" + bits + ") != uint8_t(0))";
        if (g == "float") return "uintBitsToFloat(uint(" + bits + "))";
        if (g == "float16_t") return "uint16BitsToFloat16(uint16_t(" + bits + "))";
        if (g == "double") return "uint64BitsToDouble(" + bits + ")";
        if (g == "int8_t") return "int8_t(uint8_t(" + bits + "))";
        if (g == "int16_t") return "int16_t(uint16_t(" + bits + "))";
        if (g == "int") return "int(uint(" + bits + "))";
        return g + "(" + bits + ")";
    }

    std::string shifted(const std::string& bits, long offset) {
        return offset ? "(" + bits + " << " + std::to_string(offset * 8) + ")" : bits;
    }

    std::string pack_bits(const std::string& v, QualType t, long offset, bool as_u8 = false) {
        t = bare(t);
        if (const auto* at = Ctx.getAsConstantArrayType(t)) {
            std::string out;
            long es = size_of(at->getElementType());
            for (uint64_t i = 0; i < at->getSize().getZExtValue(); i++)
                out += (i ? " | " : "") + pack_bits(v + "[" + std::to_string(i) + "]", at->getElementType(), offset + (long)i * es);
            return out.empty() ? "0ul" : "(" + out + ")";
        }
        if (const auto* rt = t->getAs<RecordType>()) {
            std::string out;
            for (auto& f : record_fields(rt->getDecl()->getDefinition()))
                out += (out.empty() ? "" : " | ") + pack_bits(v + "." + f.name, f.type, offset + f.offset, f.in_memory_bool);
            return out.empty() ? "0ul" : "(" + out + ")";
        }
        return shifted(scalar_bits(v, t, as_u8), offset);
    }

    std::string unpack_bits(const std::string& bits, QualType t, long offset, bool as_u8) {
        t = bare(t);
        std::string at_offset = offset ? "(" + bits + " >> " + std::to_string(offset * 8) + ")" : bits;
        if (const auto* at = Ctx.getAsConstantArrayType(t)) {
            std::string out = type(bare(at->getElementType())) + "[" + std::to_string(at->getSize().getZExtValue()) + "](";
            long es = size_of(at->getElementType());
            for (uint64_t i = 0; i < at->getSize().getZExtValue(); i++)
                out += (i ? ", " : "") + unpack_bits(bits, at->getElementType(), offset + (long)i * es, false);
            return out + ")";
        }
        if (const auto* rt = t->getAs<RecordType>()) {
            const RecordDecl* rd = rt->getDecl()->getDefinition();
            std::string out = record(rd) + "(";
            const auto& fields = record_fields(rd);
            for (size_t i = 0; i < fields.size(); i++)
                out += (i ? ", " : "") + unpack_bits(bits, fields[i].type, offset + fields[i].offset, fields[i].in_memory_bool);
            if (fields.empty()) out += "uint8_t(0)";
            return out + ")";
        }
        return scalar_from_bits(at_offset, t, as_u8);
    }

    // threadIdx.x & co: a pseudo-object whose result calls __fetch_builtin_<c>()
    // on a __cuda_builtin_<var>_t object
    std::string builtin_var(const PseudoObjectExpr* po) {
        const auto* ce = dyn_cast<CallExpr>(po->getResultExpr()->IgnoreImplicit());
        const MemberExpr* me = ce ? dyn_cast<MemberExpr>(ce->getCallee()->IgnoreImplicit()) : nullptr;
        if (!me) throw Unsupported("property access");
        std::string fetch = me->getMemberDecl()->getName().str();
        std::string var = record_name_of(me->getBase()->getType());
        std::string c = fetch.substr(fetch.size() - 1);
        if (c != "x" && c != "y" && c != "z") throw Unsupported("builtin " + fetch);
        if (var == "__cuda_builtin_threadIdx_t") return "gl_LocalInvocationID." + c;
        if (var == "__cuda_builtin_blockIdx_t") return "gl_WorkGroupID." + c;
        if (var == "__cuda_builtin_blockDim_t") return "gl_WorkGroupSize." + c;
        if (var == "__cuda_builtin_gridDim_t") return "gl_NumWorkGroups." + c;
        throw Unsupported("builtin " + var);
    }

    static std::string record_name_of(QualType t) {
        if (const auto* rd = t.getNonReferenceType()->getAsRecordDecl()) return rd->getNameAsString();
        return "";
    }

    std::string rvalue(const Expr* e) {
        std::string c;
        if ((isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e) || isa<UnaryExprOrTypeTraitExpr>(e) ||
             isa<ConstantExpr>(e) || isa<SubstNonTypeTemplateParmExpr>(e) || isa<SizeOfPackExpr>(e)) &&
            constant(e, c))
            return c;
        // constexpr / const globals and static members: their value
        if (const auto* ic = dyn_cast<ImplicitCastExpr>(e); ic && ic->getCastKind() == CK_LValueToRValue) {
            if (const auto* dr = dyn_cast<DeclRefExpr>(ic->getSubExpr()->IgnoreParens())) {
                const auto* vd = dyn_cast<VarDecl>(dr->getDecl());
                if (vd && !fn->vars.count(vd) && !vd->hasAttr<CUDASharedAttr>() &&
                    vd->isUsableInConstantExpressions(Ctx) && constant(e, c))
                    return c;
            }
        }
        if (const auto* pe = dyn_cast<ParenExpr>(e)) return "(" + rvalue(pe->getSubExpr()) + ")";
        if (const auto* x = dyn_cast<ExprWithCleanups>(e)) return rvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<MaterializeTemporaryExpr>(e)) return rvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<CXXBindTemporaryExpr>(e)) return rvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<CXXDefaultArgExpr>(e)) return rvalue(x->getExpr());
        if (const auto* x = dyn_cast<CXXDefaultInitExpr>(e)) return rvalue(x->getExpr());
        if (const auto* x = dyn_cast<ConstantExpr>(e)) return rvalue(x->getSubExpr());
        if (const auto* x = dyn_cast<SubstNonTypeTemplateParmExpr>(e)) return rvalue(x->getReplacement());
        if (const auto* x = dyn_cast<PseudoObjectExpr>(e)) return builtin_var(x);

        if (const auto* il = dyn_cast<IntegerLiteral>(e)) return literal_int(llvm::APSInt(il->getValue(), !il->getType()->isSignedIntegerType()), il->getType());
        if (const auto* fl = dyn_cast<FloatingLiteral>(e)) return literal_float(fl->getValueAsApproximateDouble(), fl->getType());
        if (const auto* bl = dyn_cast<CXXBoolLiteralExpr>(e)) return bl->getValue() ? "true" : "false";
        if (const auto* cl = dyn_cast<CharacterLiteral>(e)) return type(cl->getType()) + "(" + std::to_string(cl->getValue()) + ")";
        if (isa<CXXNullPtrLiteralExpr>(e) || isa<GNUNullExpr>(e)) return "0ul";
        if (const auto* ue = dyn_cast<UnaryExprOrTypeTraitExpr>(e)) {
            if (constant(ue, c)) return c;
            throw Unsupported("sizeof/alignof");
        }

        if (const auto* dr = dyn_cast<DeclRefExpr>(e)) {
            const ValueDecl* d = dr->getDecl();
            if (const auto* ec = dyn_cast<EnumConstantDecl>(d)) return literal_int(ec->getInitVal(), e->getType());
            if (const auto* vd = dyn_cast<VarDecl>(d)) {
                if (!fn->vars.count(vd) && !vd->hasAttr<CUDASharedAttr>()) {
                    if (constant(e, c)) return c;
                    if (const Expr* init = vd->getAnyInitializer()) {
                        if (vd->getType().isConstQualified() && constant(init, c)) return c;
                    }
                }
            }
            return load(lvalue(e));
        }

        if (const auto* ce = dyn_cast<CastExpr>(e)) {
            // bfloat16-style reinterpretation through a pointer is handled at the deref
            return cast(ce->getCastKind(), ce->getSubExpr(), e->getType());
        }

        if (const auto* bo = dyn_cast<BinaryOperator>(e)) return binary(bo);
        if (const auto* uo = dyn_cast<UnaryOperator>(e)) return unary(uo);
        if (const auto* co = dyn_cast<ConditionalOperator>(e)) {
            std::string t = type(e->getType());
            return "(" + rvalue_as(co->getCond(), Ctx.BoolTy) + " ? " + convert(rvalue(co->getTrueExpr()), type(co->getTrueExpr()->getType()), t) +
                   " : " + convert(rvalue(co->getFalseExpr()), type(co->getFalseExpr()->getType()), t) + ")";
        }
        if (const auto* bco = dyn_cast<BinaryConditionalOperator>(e)) {
            (void)bco;
            throw Unsupported("?: with omitted operand");
        }

        if (const auto* ctor = dyn_cast<CXXConstructExpr>(e)) return construct(ctor);
        if (const auto* sv = dyn_cast<CXXScalarValueInitExpr>(e)) return zero(sv->getType());
        if (const auto* ile = dyn_cast<InitListExpr>(e)) return init_list(ile);
        // C++20 T(a, b, c) on an aggregate: same as T{a, b, c}
        if (const auto* pl = dyn_cast<CXXParenListInitExpr>(e)) {
            auto inits = const_cast<CXXParenListInitExpr*>(pl)->getInitExprs();
            return aggregate(pl->getType(), std::vector<const Expr*>(inits.begin(), inits.end()));
        }
        if (const auto* ie = dyn_cast<ImplicitValueInitExpr>(e)) return zero(ie->getType());

        if (isa<MemberExpr>(e) || isa<ArraySubscriptExpr>(e)) return load(lvalue(e));
        if (const auto* ce = dyn_cast<CallExpr>(e)) {
            if (trivial_assign_target(ce)) return call(ce);
            QualType rt = ce->getCallReturnType(Ctx);
            if (rt->isLValueReferenceType() && !rt.getNonReferenceType().isConstQualified()) return load(lvalue(e));
            if (const FunctionDecl* fd = ce->getDirectCallee()) {
                if (fd->isInStdNamespace() && (fd->getName() == "forward" || fd->getName() == "move")) {
                    const Expr* a = ce->getArg(0);
                    return a->isGLValue() ? load(lvalue(a)) : rvalue(a);
                }
            }
            return call(ce);
        }
        if (isa<CXXThisExpr>(e)) return address(self(), "this");
        if (constant(e, c)) return c;   // anything else the compiler can evaluate
        throw Unsupported(std::string("expression ") + e->getStmtClassName());
    }

    std::string init_list(const InitListExpr* ile) {
        std::vector<const Expr*> inits;
        for (unsigned i = 0; i < ile->getNumInits(); i++) inits.push_back(ile->getInit(i));
        return aggregate(ile->getType(), inits);
    }

    // An array or struct built from its elements / fields in order (bases
    // first); missing ones are zero.
    std::string aggregate(QualType type_, const std::vector<const Expr*>& inits) {
        QualType t = bare(type_);
        struct Inits {
            const std::vector<const Expr*>& v;
            unsigned getNumInits() const { return (unsigned)v.size(); }
            const Expr* getInit(unsigned i) const { return v[i]; }
        } list{inits};
        const Inits* ile = &list;
        if (const auto* at = Ctx.getAsConstantArrayType(t)) {
            std::string elem = type(bare(at->getElementType()));
            uint64_t n = at->getSize().getZExtValue();
            std::string out = elem + "[" + std::to_string(n) + "](";
            for (uint64_t i = 0; i < n; i++) {
                if (i) out += ", ";
                out += i < ile->getNumInits() ? rvalue_as(ile->getInit(i), at->getElementType()) : zero(at->getElementType());
            }
            return out + ")";
        }
        if (const auto* rt = t->getAs<RecordType>()) {
            const RecordDecl* rd = rt->getDecl()->getDefinition();
            std::string name = record(rd);
            const auto& fields = record_fields(rd);
            std::string out = name + "(";
            // bases come first in the init list, then fields
            unsigned i = 0;
            bool first = true;
            for (auto& f : fields) {
                out += first ? "" : ", ";
                first = false;
                std::string v = i < ile->getNumInits() ? rvalue_as(ile->getInit(i), f.type) : zero(f.type);
                if (f.in_memory_bool) v = "((" + v + ") ? uint8_t(1) : uint8_t(0))";
                out += v;
                i++;
            }
            if (fields.empty()) out += "uint8_t(0)";
            return out + ")";
        }
        if (ile->getNumInits() == 0) return zero(t);
        return rvalue_as(ile->getInit(0), t);
    }

    std::string binary(const BinaryOperator* bo) {
        BinaryOperatorKind op = bo->getOpcode();
        const Expr* l = bo->getLHS();
        const Expr* r = bo->getRHS();
        if (op == BO_Comma) {
            emit_pending(rvalue(l) + ";");
            return rvalue(r);
        }
        if (bo->isAssignmentOp()) {
            LV target = lvalue(l);
            QualType lt = l->getType();
            if (lt->isPointerType() && (op == BO_AddAssign || op == BO_SubAssign)) {
                std::string step = "uint64_t(int64_t(" + rvalue(r) + ")) * " + std::to_string(size_of(lt->getPointeeType())) + "ul";
                return store(target, load(target) + (op == BO_AddAssign ? " + " : " - ") + step);
            }
            if (op == BO_Assign) return store(target, rvalue_as(r, lt));
            std::string opstr = BinaryOperator::getOpcodeStr(BinaryOperator::getOpForCompoundAssignment(op)).str();
            // compound: compute in the computation type, store back
            const auto* cao = llvm::cast<CompoundAssignOperator>(bo);
            std::string ct = type(cao->getComputationLHSType());
            std::string lhs = convert(load(target), type(lt), ct);
            std::string rhs = convert(rvalue(r), type(r->getType()), type(cao->getComputationResultType()));
            std::string v = convert("(" + lhs + " " + opstr + " " + rhs + ")", type(cao->getComputationResultType()), type(lt));
            return store(target, v);
        }
        QualType ltype = l->getType(), rtype = r->getType();
        std::string opstr = BinaryOperator::getOpcodeStr(op).str();
        if (ltype->isPointerType() && rtype->isIntegerType() && (op == BO_Add || op == BO_Sub)) {
            return "(" + rvalue(l) + (op == BO_Add ? " + " : " - ") + "uint64_t(int64_t(" + rvalue(r) + ")) * " +
                   std::to_string(size_of(ltype->getPointeeType())) + "ul)";
        }
        if (rtype->isPointerType() && ltype->isIntegerType() && op == BO_Add) {
            return "(" + rvalue(r) + " + uint64_t(int64_t(" + rvalue(l) + ")) * " +
                   std::to_string(size_of(rtype->getPointeeType())) + "ul)";
        }
        if (ltype->isPointerType() && rtype->isPointerType() && op == BO_Sub) {
            return "(int64_t(" + rvalue(l) + " - " + rvalue(r) + ") / " + std::to_string(size_of(ltype->getPointeeType())) + "l)";
        }
        if (op == BO_LAnd || op == BO_LOr) {
            return "(" + rvalue_as(l, Ctx.BoolTy) + " " + opstr + " " + rvalue_as(r, Ctx.BoolTy) + ")";
        }
        if (op == BO_Shl || op == BO_Shr) {
            return "(" + rvalue(l) + " " + opstr + " " + rvalue(r) + ")";
        }
        return "(" + rvalue(l) + " " + opstr + " " + rvalue(r) + ")";
    }

    std::string unary(const UnaryOperator* uo) {
        std::string c;
        switch (uo->getOpcode()) {
            case UO_Minus: return "(-" + rvalue(uo->getSubExpr()) + ")";
            case UO_Plus: return rvalue(uo->getSubExpr());
            case UO_LNot: return "(!" + rvalue_as(uo->getSubExpr(), Ctx.BoolTy) + ")";
            case UO_Not: return "(~" + rvalue(uo->getSubExpr()) + ")";
            case UO_Deref:
                if (pun(uo, c)) return c;
                return load(lvalue(uo));
            case UO_AddrOf: return address(lvalue(uo->getSubExpr()), "&");
            case UO_Extension: return rvalue(uo->getSubExpr());
            case UO_PreInc: case UO_PreDec: case UO_PostInc: case UO_PostDec: {
                LV t = lvalue(uo->getSubExpr());
                bool inc = uo->getOpcode() == UO_PreInc || uo->getOpcode() == UO_PostInc;
                bool pre = uo->getOpcode() == UO_PreInc || uo->getOpcode() == UO_PreDec;
                QualType ty = uo->getSubExpr()->getType();
                if (ty->isPointerType()) {
                    std::string step = std::to_string(size_of(ty->getPointeeType())) + "ul";
                    std::string assign = "(" + store(t, load(t) + (inc ? " + " : " - ") + step) + ")";
                    return pre ? assign : "(" + assign + (inc ? " - " : " + ") + step + ")";
                }
                if (t.u8bool || bare(ty)->isBooleanType()) throw Unsupported("++ on bool");
                return pre ? "(" + std::string(inc ? "++" : "--") + place(t) + ")" : "(" + place(t) + (inc ? "++" : "--") + ")";
            }
            default: throw Unsupported("unary operator");
        }
    }

    // ======================================================================
    //  Calls
    // ======================================================================

    static bool from_runtime(const FunctionDecl* fd, const SourceManager& sm) {
        SourceLocation loc = sm.getSpellingLoc(fd->getLocation());
        std::string file = sm.getFilename(loc).str();
        return file.find("ops/vulkan/runtime.hpp") != std::string::npos;
    }

    static bool is_intrinsic(const std::string& n) {
        static const std::set<std::string> names = {
            "__syncthreads", "__syncwarp", "__threadfence", "__threadfence_block",
            "atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor",
            "atomicCAS", "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync", "__shfl",
            "__shfl_up", "__shfl_down", "__shfl_xor", "__any_sync", "__all_sync", "__ballot_sync", "__ldg",
            "__float_as_int", "__float_as_uint", "__int_as_float", "__uint_as_float", "__popc", "__clz", "__ffs",
            "__expf", "__logf", "__sinf", "__cosf", "__powf", "__fdividef", "__frsqrt_rn", "__saturatef", "__fmaf_rn",
        };
        return names.count(n) != 0;
    }

    std::string math(const std::string& raw, const CallExpr* ce) {
        std::string n = raw;
        if (n.rfind("__builtin_", 0) == 0) n = n.substr(10);
        static const std::map<std::string, std::string> fns = {
            {"exp", "exp"}, {"expf", "exp"}, {"__expf", "exp"}, {"exp2", "exp2"}, {"exp2f", "exp2"},
            {"log", "log"}, {"logf", "log"}, {"__logf", "log"}, {"log2", "log2"}, {"log2f", "log2"},
            {"log10", "vk_log10"}, {"log10f", "vk_log10"},
            {"sqrt", "sqrt"}, {"sqrtf", "sqrt"}, {"rsqrt", "inversesqrt"}, {"rsqrtf", "inversesqrt"},
            {"__frsqrt_rn", "inversesqrt"},
            {"pow", "pow"}, {"powf", "pow"}, {"__powf", "pow"},
            {"sin", "sin"}, {"sinf", "sin"}, {"__sinf", "sin"}, {"cos", "cos"}, {"cosf", "cos"}, {"__cosf", "cos"},
            {"tan", "tan"}, {"tanf", "tan"}, {"tanh", "tanh"}, {"tanhf", "tanh"},
            {"sinh", "sinh"}, {"sinhf", "sinh"}, {"cosh", "cosh"}, {"coshf", "cosh"},
            {"atan", "atan"}, {"atanf", "atan"}, {"atan2", "atan"}, {"atan2f", "atan"},
            {"asin", "asin"}, {"asinf", "asin"}, {"acos", "acos"}, {"acosf", "acos"},
            {"fabs", "abs"}, {"fabsf", "abs"}, {"abs", "abs"}, {"labs", "abs"}, {"llabs", "abs"},
            {"floor", "floor"}, {"floorf", "floor"}, {"ceil", "ceil"}, {"ceilf", "ceil"},
            {"round", "round"}, {"roundf", "round"}, {"trunc", "trunc"}, {"truncf", "trunc"},
            {"fmin", "min"}, {"fminf", "min"}, {"fmax", "max"}, {"fmaxf", "max"}, {"min", "min"}, {"max", "max"},
            {"fma", "fma"}, {"fmaf", "fma"}, {"__fmaf_rn", "fma"},
            {"erf", "vk_erf"}, {"erff", "vk_erf"}, {"isnan", "isnan"}, {"isinf", "isinf"},
            {"copysign", "vk_copysign"}, {"copysignf", "vk_copysign"}, {"__saturatef", "vk_saturate"},
            {"__fdividef", "vk_div"},
        };
        if (n == "inf" || n == "inff" || n == "huge_val" || n == "huge_valf") return "uintBitsToFloat(0x7f800000u)";
        if (n == "nan" || n == "nanf") return "uintBitsToFloat(0x7fc00000u)";
        if (n == "expect") return rvalue(ce->getArg(0));
        auto f = fns.find(n);
        if (f == fns.end()) return "";
        std::string g = f->second;
        if (g.rfind("vk_", 0) == 0) math_helpers();
        std::string ret = type(ce->getType());
        // transcendental functions exist for float only
        bool float_only = !(g == "abs" || g == "min" || g == "max" || g == "floor" || g == "ceil" || g == "round" ||
                            g == "trunc" || g == "sqrt" || g == "fma" || g == "isnan" || g == "isinf");
        std::string out = g + "(";
        for (unsigned i = 0; i < ce->getNumArgs(); i++) {
            std::string a = rvalue(ce->getArg(i));
            std::string at = type(ce->getArg(i)->getType());
            if (float_only && at != "float") a = "float(" + a + ")";
            else if (!float_only && at != ret && ret != "bool") a = convert(a, at, ret);
            out += (i ? ", " : "") + a;
        }
        out += ")";
        if (float_only && ret != "float" && ret != "bool") out = ret + "(" + out + ")";
        return out;
    }

    void math_helpers() {
        if (!helpers.insert("math").second) return;
        helper_decls +=
            "float vk_log10(float x) { return log(x) * 0.43429448190325176; }\n"
            "float vk_copysign(float a, float b) { return b < 0.0 ? -abs(a) : abs(a); }\n"
            "float vk_saturate(float x) { return clamp(x, 0.0, 1.0); }\n"
            "float vk_div(float a, float b) { return a / b; }\n"
            "float vk_erf(float x) {\n"
            "    float z = abs(x);\n"
            "    float t = 1.0 / (1.0 + 0.5 * z);\n"
            "    float r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +\n"
            "              t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +\n"
            "              t * (-0.82215223 + t * 0.17087277)))))))));\n"
            "    return x >= 0.0 ? 1.0 - r : r - 1.0;\n"
            "}\n";
    }

    // shuffles: subgroup op when the subgroup covers `width`, else through shared memory
    std::string shuffle(const std::string& kind, const CallExpr* ce, bool has_mask) {
        unsigned vi = has_mask ? 1 : 0;
        const Expr* v = ce->getArg(vi);
        std::string t = type(v->getType());
        if (t != "float" && t != "int" && t != "uint" && t != "double" && t != "int64_t" && t != "uint64_t")
            throw Unsupported("shuffle of " + t);
        use_subgroup = true;
        use_shuffle_scratch = true;
        std::string name = "vk_shfl_" + kind + "_" + sanitize(t);
        if (helpers.insert(name).second) {
            if (helpers.insert("scratch").second) {
                shared_decls += "shared uint vk_scratch[gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z * 2];\n";
            }
            bool wide = t == "double" || t == "int64_t" || t == "uint64_t";
            std::string to_bits = t == "float" ? "uvec2(floatBitsToUint(v), 0u)" : t == "int" ? "uvec2(uint(v), 0u)"
                                : t == "uint" ? "uvec2(v, 0u)" : t == "double" ? "unpackDouble2x32(v)"
                                : "unpack32(uint64_t(v))";
            std::string from_bits = t == "float" ? "uintBitsToFloat(b.x)" : t == "int" ? "int(b.x)"
                                  : t == "uint" ? "b.x" : t == "double" ? "packDouble2x32(b)"
                                  : t + "(pack64(b))";
            (void)wide;
            // source lane within the width-sized segment; own value when out of range (CUDA semantics)
            std::string src = kind == "down" ? "seg + d" : kind == "up" ? "int(seg) - int(d)" : kind == "xor" ? "seg ^ d" : "d % w";
            std::string ok = kind == "down" ? "seg + d < w" : kind == "up" ? "seg >= d" : "true";
            std::string sub = kind == "down" ? "subgroupShuffleDown(v, d)" : kind == "up" ? "subgroupShuffleUp(v, d)"
                            : kind == "xor" ? "subgroupShuffleXor(v, d)" : "subgroupShuffle(v, (gl_SubgroupInvocationID / w) * w + d % w)";
            helper_decls +=
                t + " " + name + "(" + t + " v, uint d, uint w) {\n"
                "    uint lane = gl_LocalInvocationIndex;\n"
                "    uint seg = lane % w;\n"
                "    if (gl_SubgroupSize >= w) {\n"
                "        " + t + " r = " + sub + ";\n"
                "        return (" + ok + ") ? r : v;\n"
                "    }\n"
                "    uvec2 bits = " + to_bits + ";\n"
                "    vk_scratch[2u * lane] = bits.x;\n"
                "    vk_scratch[2u * lane + 1u] = bits.y;\n"
                "    memoryBarrierShared();\n"
                "    barrier();\n"
                "    uint s = (lane / w) * w + uint(" + src + ");\n"
                "    uvec2 b = uvec2(vk_scratch[2u * s], vk_scratch[2u * s + 1u]);\n"
                "    " + t + " r = (" + ok + ") ? " + from_bits + " : v;\n"
                "    barrier();\n"
                "    return r;\n"
                "}\n";
        }
        std::string d = rvalue(ce->getArg(vi + 1));
        std::string w = ce->getNumArgs() > vi + 2 ? rvalue(ce->getArg(vi + 2)) : "32";
        return name + "(" + rvalue(v) + ", uint(" + d + "), uint(" + w + "))";
    }

    std::string atomic(const std::string& n, const CallExpr* ce) {
        const Expr* p = ce->getArg(0)->IgnoreParenImpCasts();
        QualType t = ce->getArg(0)->getType()->getPointeeType();
        std::string target;
        if (const auto* ao = dyn_cast<UnaryOperator>(p); ao && ao->getOpcode() == UO_AddrOf) {
            target = place(lvalue(ao->getSubExpr()));
        } else {
            LV lv;
            lv.kind = LV::Mem;
            lv.text = rvalue(ce->getArg(0));
            lv.type = t;
            target = place(lv);
        }
        std::string ty = type(t);
        if (ty == "float") use_atomic_float = true;
        if (ty == "int64_t" || ty == "uint64_t") use_atomic_int64 = true;
        std::string v = rvalue_as(ce->getArg(1), t);
        if (n == "atomicAdd") return "atomicAdd(" + target + ", " + v + ")";
        if (n == "atomicSub") return "atomicAdd(" + target + ", -(" + v + "))";
        if (n == "atomicExch") return "atomicExchange(" + target + ", " + v + ")";
        if (n == "atomicMin") return "atomicMin(" + target + ", " + v + ")";
        if (n == "atomicMax") return "atomicMax(" + target + ", " + v + ")";
        if (n == "atomicAnd") return "atomicAnd(" + target + ", " + v + ")";
        if (n == "atomicOr") return "atomicOr(" + target + ", " + v + ")";
        if (n == "atomicXor") return "atomicXor(" + target + ", " + v + ")";
        if (n == "atomicCAS") return "atomicCompSwap(" + target + ", " + v + ", " + rvalue_as(ce->getArg(2), t) + ")";
        throw Unsupported(n);
    }

    std::string intrinsic(const FunctionDecl* fd, const CallExpr* ce) {
        std::string n = fd->getNameAsString();
        if (n == "__syncthreads") return "memoryBarrierShared(), memoryBarrierBuffer(), barrier()";
        if (n == "__syncwarp") { use_subgroup = true; return "subgroupBarrier()"; }
        if (n == "__threadfence") return "memoryBarrier()";
        if (n == "__threadfence_block") return "groupMemoryBarrier()";
        if (n.rfind("atomic", 0) == 0) return atomic(n, ce);
        if (n == "__shfl_down_sync") return shuffle("down", ce, true);
        if (n == "__shfl_up_sync") return shuffle("up", ce, true);
        if (n == "__shfl_xor_sync") return shuffle("xor", ce, true);
        if (n == "__shfl_sync") return shuffle("idx", ce, true);
        if (n == "__shfl_down") return shuffle("down", ce, false);
        if (n == "__shfl_up") return shuffle("up", ce, false);
        if (n == "__shfl_xor") return shuffle("xor", ce, false);
        if (n == "__shfl") return shuffle("idx", ce, false);
        if (n == "__any_sync") { use_subgroup = true; return "(subgroupAny(" + rvalue_as(ce->getArg(1), Ctx.BoolTy) + ") ? 1 : 0)"; }
        if (n == "__all_sync") { use_subgroup = true; return "(subgroupAll(" + rvalue_as(ce->getArg(1), Ctx.BoolTy) + ") ? 1 : 0)"; }
        if (n == "__ballot_sync") { use_subgroup = true; return "subgroupBallot(" + rvalue_as(ce->getArg(1), Ctx.BoolTy) + ").x"; }
        if (n == "__ldg") {
            LV lv;
            lv.kind = LV::Mem;
            lv.text = rvalue(ce->getArg(0));
            lv.type = ce->getType();
            return load(lv);
        }
        if (n == "__float_as_int") return "floatBitsToInt(" + rvalue(ce->getArg(0)) + ")";
        if (n == "__float_as_uint") return "floatBitsToUint(" + rvalue(ce->getArg(0)) + ")";
        if (n == "__int_as_float") return "intBitsToFloat(" + rvalue(ce->getArg(0)) + ")";
        if (n == "__uint_as_float") return "uintBitsToFloat(" + rvalue(ce->getArg(0)) + ")";
        if (n == "__popc") return "bitCount(" + rvalue(ce->getArg(0)) + ")";
        if (n == "__clz") return "(31 - findMSB(" + rvalue(ce->getArg(0)) + "))";
        if (n == "__ffs") return "(findLSB(" + rvalue(ce->getArg(0)) + ") + 1)";
        std::string m = math(n, ce);
        if (!m.empty()) return m;
        throw Unsupported("device intrinsic " + n);
    }

    // How each reference parameter (and the object, first) is passed.
    enum Bind : char { ByAddress = 'M', ByInout = 'R', ByValue = 'V' };

    // a = b on a trivially copyable struct (an implicit operator=)
    static const Expr* trivial_assign_target(const CallExpr* ce) {
        const auto* oc = dyn_cast<CXXOperatorCallExpr>(ce);
        if (!oc) return nullptr;
        const auto* md = dyn_cast_or_null<CXXMethodDecl>(oc->getDirectCallee());
        if (md && (md->isCopyAssignmentOperator() || md->isMoveAssignmentOperator()) && md->isTrivial()) return oc->getArg(0);
        return nullptr;
    }

    std::string call(const CallExpr* ce) {
        const FunctionDecl* fd = ce->getDirectCallee();
        if (!fd) throw Unsupported("indirect call");
        std::string n = fd->getNameAsString();
        if (from_runtime(fd, SM) || is_intrinsic(n)) return intrinsic(fd, ce);
        if (fd->getBuiltinID() || !fd->hasBody()) {
            if (n == "printf" || n == "__assert_fail" || n == "__builtin_unreachable" || n == "__builtin_assume") return "";
            std::string m = math(n, ce);
            if (!m.empty()) return m;
            throw Unsupported("call to " + fd->getQualifiedNameAsString() + " (no body)");
        }
        if (fd->isInStdNamespace() || fd->getDeclContext()->isTranslationUnit()) {
            std::string m = math(n, ce);
            if (!m.empty() && fd->getNumParams() && fd->getParamDecl(0)->getType()->isArithmeticType()) return m;
        }

        // implicit object
        const Expr* object = nullptr;
        const CXXMethodDecl* md = dyn_cast<CXXMethodDecl>(fd);
        unsigned first_arg = 0;
        if (const auto* mc = dyn_cast<CXXMemberCallExpr>(ce)) object = mc->getImplicitObjectArgument();
        else if (const auto* oc = dyn_cast<CXXOperatorCallExpr>(ce); oc && md && !md->isStatic()) {
            object = oc->getArg(0);
            first_arg = 1;
        }
        if (md && !md->isStatic() && !object) throw Unsupported("member call without object");

        // trivial copy / move assignment of a struct
        if (md && (md->isCopyAssignmentOperator() || md->isMoveAssignmentOperator()) && md->isTrivial()) {
            return store(lvalue(object), rvalue(ce->getArg(first_arg)));
        }

        std::string binding;
        std::vector<std::string> args;
        const Lambda* lambda = nullptr;
        bind_arguments(ce, fd, md, object, first_arg, binding, args, lambda);
        lambda_for_function = lambda;
        std::string name;
        try {
            name = function(fd, binding);
        } catch (...) {
            lambda_for_function = nullptr;
            throw;
        }
        lambda_for_function = nullptr;
        std::string out = name + "(";
        for (size_t i = 0; i < args.size(); i++) out += (i ? ", " : "") + args[i];
        return out + ")";
    }

    // Arguments of a call and how each is passed (the function variant):
    // the implicit object / lambda captures first, then the parameters.
    void bind_arguments(const CallExpr* ce, const FunctionDecl* fd, const CXXMethodDecl* md, const Expr* object,
                        unsigned first_arg, std::string& binding, std::vector<std::string>& args,
                        const Lambda*& lambda) {
        if (md && md->getParent()->isLambda()) {
            lambda = &lambda_of(md, object);
            for (const LV& c : lambda->captures) {
                if (c.kind == LV::Mem) { binding += ByAddress; args.push_back(c.text); }
                else { binding += ByInout; args.push_back(writable_place(c)); }
            }
        } else if (md && !md->isStatic()) {
            bool arrow = false;
            if (const auto* mc = dyn_cast<CXXMemberCallExpr>(ce)) {
                if (const auto* me = dyn_cast<MemberExpr>(mc->getCallee()->IgnoreParens())) arrow = me->isArrow();
            }
            if (arrow) {
                if (isa<CXXThisExpr>(object->IgnoreParenImpCasts())) {
                    LV s = self();
                    if (s.kind == LV::Mem) { binding += ByAddress; args.push_back(s.text); }
                    else if (md->isConst()) { binding += ByValue; args.push_back(load(s)); }
                    else { binding += ByInout; args.push_back(writable_place(s)); }
                } else {
                    binding += ByAddress;
                    args.push_back(rvalue(object));
                }
            } else if (object->isPRValue()) {
                binding += ByValue;
                args.push_back(rvalue(object));
            } else {
                // a reference the method returns into a by-value object fails
                // when its body is translated (address of a local)
                LV o = lvalue(object);
                if (o.kind == LV::Mem) { binding += ByAddress; args.push_back(o.text); }
                else if (md->isConst()) { binding += ByValue; args.push_back(load(o)); }
                else { binding += ByInout; args.push_back(writable_place(o)); }
            }
        }
        for (unsigned i = 0; i < fd->getNumParams(); i++) {
            const ParmVarDecl* p = fd->getParamDecl(i);
            unsigned ai = i + first_arg;
            if (ai >= ce->getNumArgs()) throw Unsupported("missing argument");
            const Expr* a = ce->getArg(ai);
            QualType pt = p->getType();
            if (pt->isLValueReferenceType() && !pt.getNonReferenceType().isConstQualified()) {
                LV lv = lvalue(a);
                if (lv.kind == LV::Mem) { binding += ByAddress; args.push_back(lv.text); }
                else { binding += ByInout; args.push_back(writable_place(lv)); }
            } else {
                binding += ByValue;
                args.push_back(rvalue_as(a, pt.getNonReferenceType()));
            }
        }
    }

    // The object expression and first argument index of a call.
    static const Expr* call_object(const CallExpr* ce, const CXXMethodDecl* md, unsigned& first_arg) {
        first_arg = 0;
        if (const auto* mc = dyn_cast<CXXMemberCallExpr>(ce)) return mc->getImplicitObjectArgument();
        if (const auto* oc = dyn_cast<CXXOperatorCallExpr>(ce); oc && md && !md->isStatic()) {
            first_arg = 1;
            return oc->getArg(0);
        }
        return nullptr;
    }

    // A method whose every return is `*this` (operator=, operator+= ...): its
    // GLSL function updates self and returns nothing, and a call's result is
    // the object it was called on.
    static bool is_this(const Expr* e) {
        if (!e) return false;
        e = e->IgnoreImplicit()->IgnoreParens();
        const auto* uo = dyn_cast<UnaryOperator>(e);
        return uo && uo->getOpcode() == UO_Deref && isa<CXXThisExpr>(uo->getSubExpr()->IgnoreParenImpCasts());
    }
    static bool all_returns_this(const Stmt* s, int& count) {
        if (!s || isa<LambdaExpr>(s)) return true;
        if (const auto* rs = dyn_cast<ReturnStmt>(s)) {
            if (!is_this(rs->getRetValue())) return false;
            count++;
            return true;
        }
        for (const Stmt* c : s->children()) if (!all_returns_this(c, count)) return false;
        return true;
    }
    static bool returns_this(const FunctionDecl* fd) {
        if (const FunctionDecl* d = fd->getDefinition()) fd = d;
        const auto* md = dyn_cast<CXXMethodDecl>(fd);
        if (!md || md->isStatic() || !fd->hasBody() || isa<CXXConstructorDecl>(fd)) return false;
        QualType rt = fd->getReturnType();
        if (!rt->isLValueReferenceType()) return false;
        int count = 0;
        return all_returns_this(fd->getBody(), count) && count > 0;
    }

    // `e` as a call of `fd` itself (a tail call when returned).
    static const CallExpr* self_call(const Expr* e, const FunctionDecl* fd) {
        if (!e || !fd) return nullptr;
        e = e->IgnoreImplicit()->IgnoreParens();
        if (const auto* ew = dyn_cast<ExprWithCleanups>(e)) e = ew->getSubExpr()->IgnoreImplicit()->IgnoreParens();
        const auto* ce = dyn_cast<CallExpr>(e);
        if (!ce || !ce->getDirectCallee()) return nullptr;
        const FunctionDecl* callee = ce->getDirectCallee();
        if (const FunctionDecl* d = callee->getDefinition()) callee = d;
        return callee == fd ? ce : nullptr;
    }

    // Whether `s` returns a call of `fd` (tail recursion).  A tail call inside
    // a loop of the body can't jump back to the top, so it is refused.
    static bool has_tail_self_call(const Stmt* s, const FunctionDecl* fd, bool in_loop) {
        if (!s) return false;
        if (const auto* rs = dyn_cast<ReturnStmt>(s)) {
            if (!self_call(rs->getRetValue(), fd)) return false;
            if (in_loop) throw Unsupported("recursion in " + fd->getNameAsString() + " (tail call inside a loop)");
            return true;
        }
        if (isa<LambdaExpr>(s)) return false;
        bool loop = isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s) || isa<CXXForRangeStmt>(s);
        bool found = false;
        for (const Stmt* c : s->children()) found = has_tail_self_call(c, fd, in_loop || loop) || found;
        return found;
    }

    // return f(args) inside f: rebind self and the parameters, go round again.
    std::string tail_call(const CallExpr* ce) {
        const FunctionDecl* fd = ce->getDirectCallee();
        const auto* md = dyn_cast<CXXMethodDecl>(fd);
        unsigned first_arg = 0;
        const Expr* object = (md && !md->isStatic()) ? call_object(ce, md, first_arg) : nullptr;
        std::string binding;
        std::vector<std::string> args;
        const Lambda* lambda = nullptr;
        bind_arguments(ce, fd, md, object, first_arg, binding, args, lambda);
        if (binding != fn->binding || args.size() != fn->slots.size())
            throw Unsupported("recursion in " + fd->getNameAsString() + " (arguments passed differently)");
        std::vector<std::pair<std::string, std::string>> assigns;   // slot name, temp
        for (size_t i = 0; i < args.size(); i++) {
            std::string d = fn->slots[i];
            std::string slot = d.substr(d.find_last_of(' ') + 1);
            if (binding[i] == ByInout) {
                if (args[i] != slot) throw Unsupported("recursion in " + fd->getNameAsString() + " (reference to another local)");
                continue;
            }
            if (d.find('[') != std::string::npos) throw Unsupported("recursion with an array parameter");
            std::string tmp = fresh("next");
            emit_pending(d.substr(0, d.size() - slot.size()) + tmp + " = " + args[i] + ";");
            assigns.push_back({slot, tmp});
        }
        std::string out;
        for (auto& a : assigns) out += a.first + " = " + a.second + "; ";
        return out + "continue;";
    }

    // Bind a lambda's captures where it is made.
    const Lambda& make_lambda(const LambdaExpr* le) {
        Lambda l;
        l.expr = le;
        l.made_in = fn;
        for (const LambdaCapture& c : le->captures()) {
            LV lv;
            if (c.capturesThis()) {
                lv = self();
            } else if (c.capturesVariable()) {
                const auto* v = dyn_cast<VarDecl>(c.getCapturedVar());
                auto found = v ? fn->vars.find(v) : fn->vars.end();
                if (found == fn->vars.end()) throw Unsupported("lambda capture of a non-local");
                lv = found->second;
                if (c.getCaptureKind() == LCK_ByCopy) {        // value at creation
                    std::string name = fresh(v->getNameAsString());
                    emit_pending(decl(lv.type, name) + " = " + load(lv) + ";");
                    lv.kind = LV::Reg;
                    lv.text = name;
                    lv.u8bool = false;
                }
            } else {
                throw Unsupported("lambda capture kind");
            }
            l.captures.push_back(lv);
        }
        return lambdas[le->getLambdaClass()] = l;
    }

    // The lambda a call operator belongs to (made in this function).
    const Lambda& lambda_of(const CXXMethodDecl* md, const Expr* object) {
        if (object) {
            if (const auto* le = dyn_cast<LambdaExpr>(object->IgnoreImplicit())) return make_lambda(le);   // [..](..){..}(args)
        }
        auto found = lambdas.find(md->getParent());
        if (found == lambdas.end()) throw Unsupported("call of a lambda not made in this kernel code");
        if (found->second.made_in != fn) throw Unsupported("lambda called outside the function that made it (pass values, not lambdas)");
        return found->second;
    }

    // CXXConstructExpr → value
    std::string construct(const CXXConstructExpr* ce) {
        const CXXConstructorDecl* cd = ce->getConstructor();
        QualType t = ce->getType();
        if (Ctx.getAsConstantArrayType(bare(t))) {
            if (cd->isTrivial() || cd->isDefaultConstructor()) return zero(t);
            throw Unsupported("array of objects with constructors");
        }
        if ((cd->isCopyOrMoveConstructor() && cd->isTrivial()) ||
            (cd->isCopyOrMoveConstructor() && cd->isDefaulted())) {
            const Expr* a = ce->getArg(0);
            return a->isGLValue() ? load(lvalue(a)) : rvalue(a);
        }
        if (cd->isTrivial()) return zero(t);
        if (!cd->hasBody() && !cd->isDefaulted()) throw Unsupported("constructor without body");
        // constructor function returning the object
        std::string binding;
        std::vector<std::string> args;
        for (unsigned i = 0; i < cd->getNumParams(); i++) {
            const ParmVarDecl* p = cd->getParamDecl(i);
            const Expr* a = ce->getArg(i);
            QualType pt = p->getType();
            if (pt->isLValueReferenceType() && !pt.getNonReferenceType().isConstQualified()) {
                LV lv = lvalue(a);
                if (lv.kind == LV::Mem) { binding += ByAddress; args.push_back(lv.text); }
                else { binding += ByInout; args.push_back(writable_place(lv)); }
            } else {
                binding += ByValue;
                args.push_back(rvalue_as(a, pt.getNonReferenceType()));
            }
        }
        std::string name = function(cd, binding);
        std::string out = name + "(";
        for (size_t i = 0; i < args.size(); i++) out += (i ? ", " : "") + args[i];
        return out + ")";
    }

    // ======================================================================
    //  Functions
    // ======================================================================

    std::string function(const FunctionDecl* fd, const std::string& binding) {
        if (const FunctionDecl* def = fd->getDefinition()) fd = def;
        auto key = std::make_pair(fd, binding);
        auto found = functions.find(key);
        if (found != functions.end()) return found->second;
        if (function_in_progress.count(fd)) throw Unsupported("recursion in " + fd->getNameAsString());
        function_in_progress.insert(fd);

        std::string name = "F" + std::to_string(counter++) + "_" + sanitize(fd->getNameAsString());

        Fn f;
        Fn* outer = fn;
        auto saved_pending = std::move(pending);
        pending.clear();
        fn = &f;

        const auto* md = dyn_cast<CXXMethodDecl>(fd);
        const auto* ctor = dyn_cast<CXXConstructorDecl>(fd);
        std::vector<std::string> params;
        size_t bi = 0;
        QualType object_type;
        if (md) object_type = Ctx.getRecordType(md->getParent());

        const Lambda* lambda = lambda_for_function;
        lambda_for_function = nullptr;   // nested calls in the body are their own
        if (lambda) {
            // captures first: `this` becomes self, variables become parameters
            const auto& caps = lambda->expr->captures();
            size_t ci = 0;
            for (const LambdaCapture& c : caps) {
                const LV& outer_lv = lambda->captures[ci];
                char b = binding[bi++];
                std::string pname = "c" + std::to_string(ci) + "_" +
                                    (c.capturesThis() ? std::string("this") : sanitize(c.getCapturedVar()->getNameAsString()));
                LV lv;
                lv.type = outer_lv.type;
                if (b == ByAddress) {
                    params.push_back("uint64_t " + pname);
                    lv.kind = LV::Mem;
                } else {
                    params.push_back("inout " + decl(outer_lv.type, pname));
                    lv.kind = LV::Reg;
                    lv.u8bool = outer_lv.u8bool;
                }
                lv.text = pname;
                if (c.capturesThis()) { f.has_self = true; f.self = lv; }
                else f.vars[c.getCapturedVar()] = lv;
                ci++;
            }
        } else if (md && !md->isStatic() && !ctor) {
            char b = binding[bi++];
            f.has_self = true;
            f.self.type = object_type;
            if (b == ByAddress) {
                params.push_back("uint64_t self_addr");
                f.slots.push_back("uint64_t self_addr");
                f.self.kind = LV::Mem;
                f.self.text = "self_addr";
            } else {
                params.push_back(std::string(b == ByInout ? "inout " : "") + type(object_type) + " self");
                f.slots.push_back(type(object_type) + " self");
                f.self.kind = LV::Reg;
                f.self.text = "self";
            }
        }
        for (unsigned i = 0; i < fd->getNumParams(); i++) {
            const ParmVarDecl* p = fd->getParamDecl(i);
            char b = binding[bi++];
            std::string pname = "p" + std::to_string(i) + "_" + sanitize(p->getNameAsString());
            LV lv;
            lv.type = p->getType().getNonReferenceType();
            if (b == ByAddress) {
                params.push_back("uint64_t " + pname);
                f.slots.push_back("uint64_t " + pname);
                lv.kind = LV::Mem;
            } else {
                params.push_back(std::string(b == ByInout ? "inout " : "") + decl(p->getType().getNonReferenceType(), pname));
                f.slots.push_back(decl(p->getType().getNonReferenceType(), pname));
                lv.kind = LV::Reg;
            }
            lv.text = pname;
            f.vars[p] = lv;
        }
        if (!lambda) {
            f.decl = fd;
            f.binding = binding;
        }

        std::string ret;
        std::string body;
        QualType rt = fd->getReturnType();
        if (ctor) {
            // build the object in a local, run the initialisers and the body
            ret = type(object_type);
            f.has_self = true;
            f.self.kind = LV::Reg;
            f.self.text = "self";
            f.self.type = object_type;
            body += "    " + type(object_type) + " self = " + zero(object_type) + ";\n";
            for (const CXXCtorInitializer* init : ctor->inits()) {
                std::string line;
                if (init->isBaseInitializer()) {
                    const CXXRecordDecl* base = init->getBaseClass()->getAsCXXRecordDecl();
                    if (base->isEmpty()) continue;
                    LV b = base_of(f.self, ctor->getParent(), base, QualType(init->getBaseClass(), 0));
                    line = store(b, rvalue(init->getInit()));
                } else if (init->isMemberInitializer()) {
                    const FieldDecl* fdl = init->getMember();
                    LV target;
                    for (auto& fi : record_fields(ctor->getParent())) {
                        if (fi.field && fi.field->getCanonicalDecl() == fdl->getCanonicalDecl()) {
                            target = field_of(f.self, fi.name, fi.offset, fdl->getType(), fi.in_memory_bool);
                        }
                    }
                    const Expr* ie = init->getInit();
                    if (isa<ImplicitValueInitExpr>(ie)) continue;
                    if (const auto* pi = dyn_cast<ParenListExpr>(ie)) {
                        if (pi->getNumExprs() != 1) throw Unsupported("member initialiser list");
                        ie = pi->getExpr(0);
                    }
                    if (Ctx.getAsConstantArrayType(bare(fdl->getType()))) {
                        // array member: copied from another array (the implicit
                        // copy / move constructors' ArrayInitLoopExpr) or braced
                        const Expr* src = ie->IgnoreImplicit();
                        if (const auto* al = dyn_cast<ArrayInitLoopExpr>(src)) src = al->getCommonExpr()->getSourceExpr();
                        src = src->IgnoreImplicit();
                        if (const auto* il = dyn_cast<InitListExpr>(src)) line = store(target, init_list(il));
                        else line = copy_array(target, lvalue_or_temp(src), fdl->getType());
                    } else {
                        line = store(target, rvalue_as(ie, fdl->getType()));
                    }
                } else {
                    throw Unsupported("delegating constructor");
                }
                for (auto& pnd : pending) body += "    " + pnd + "\n";
                pending.clear();
                body += "    " + line + ";\n";
            }
            if (fd->hasBody()) body += stmt(fd->getBody(), 1);
            body += "    return self;\n";
        } else {
            if (returns_this(fd)) {
                ret = "void";
                f.returns_self = true;
            } else if (rt->isLValueReferenceType() && !rt.getNonReferenceType().isConstQualified()) {
                ret = "uint64_t";
                f.returns_address = true;
            } else {
                ret = type(rt.getNonReferenceType());
            }
            f.return_type = rt;
            if (f.decl && has_tail_self_call(fd->getBody(), fd, false)) {
                f.tail_loop = true;
                body = "    for (;;) {\n" + stmt(fd->getBody(), 2) +
                       (rt->isVoidType() ? "        return;\n" : "") + "    }\n";
            } else {
                body = stmt(fd->getBody(), 1);
            }
        }

        std::string sig = ret + " " + name + "(";
        for (size_t i = 0; i < params.size(); i++) sig += (i ? ", " : "") + params[i];
        sig += ")";
        function_defs += sig + " {\n" + body + "}\n\n";

        fn = outer;
        pending = std::move(saved_pending);
        function_in_progress.erase(fd);
        return functions[key] = name;
    }

    // ======================================================================
    //  Statements
    // ======================================================================

    static std::string ind(int n) { return std::string(n * 4, ' '); }

    // An expression statement with its pending temporaries.
    std::string with_pending(int depth, const std::string& line) {
        std::string out;
        for (auto& p : pending) out += ind(depth) + p + "\n";
        pending.clear();
        return out + (line.empty() ? "" : ind(depth) + line + "\n");
    }

    std::string var_decl(const VarDecl* vd, int depth) {
        if (vd->hasAttr<CUDASharedAttr>()) {
            shared(vd);
            return "";
        }
        if (vd->isStaticLocal()) throw Unsupported("static local variable " + vd->getNameAsString());
        QualType t = vd->getType();
        const Expr* init = vd->getInit();
        if (init) {
            if (const auto* le = dyn_cast<LambdaExpr>(init->IgnoreImplicit())) {
                make_lambda(le);
                return with_pending(depth, "");
            }
        }
        if (t->isReferenceType()) {
            if (!init) throw Unsupported("reference without initialiser");
            bool const_ref = t.getNonReferenceType().isConstQualified() || t->isRValueReferenceType();
            if (const_ref) {
                std::string name = fresh(vd->getNameAsString());
                std::string v = rvalue_as(init, t.getNonReferenceType());
                LV lv;
                lv.kind = LV::Reg;
                lv.text = name;
                lv.type = t.getNonReferenceType();
                fn->vars[vd] = lv;
                return with_pending(depth, decl(t.getNonReferenceType(), name) + " = " + v + ";");
            }
            LV target = lvalue(init);
            if (target.kind == LV::Mem) {
                std::string name = fresh(vd->getNameAsString());
                LV lv = target;
                lv.text = name;
                fn->vars[vd] = lv;
                return with_pending(depth, "uint64_t " + name + " = " + target.text + ";");
            }
            fn->vars[vd] = target;   // alias of a local
            return with_pending(depth, "");
        }
        std::string name = fresh(vd->getNameAsString());
        LV lv;
        lv.kind = LV::Reg;
        lv.text = name;
        lv.type = t;
        fn->vars[vd] = lv;
        std::string d = decl(t, name);
        if (!init) return with_pending(depth, d + ";");
        if (const auto* ce = dyn_cast<CXXConstructExpr>(init->IgnoreImplicit())) {
            if (ce->getConstructor()->isTrivial() && ce->getNumArgs() == 0) return with_pending(depth, d + " = " + zero(t) + ";");
        }
        if (Ctx.getAsConstantArrayType(bare(t)) && isa<InitListExpr>(init->IgnoreImplicit()))
            return with_pending(depth, d + " = " + init_list(llvm::cast<InitListExpr>(init->IgnoreImplicit())) + ";");
        std::string v = rvalue_as(init, t);
        return with_pending(depth, d + " = " + v + ";");
    }

    std::string stmt(const Stmt* s, int depth) {
        if (!s) return "";
        if (const auto* cs = dyn_cast<CompoundStmt>(s)) {
            std::string out;
            for (const Stmt* c : cs->body()) out += stmt(c, depth);
            return out;
        }
        if (const auto* as = dyn_cast<AttributedStmt>(s)) return stmt(as->getSubStmt(), depth);
        if (const auto* ds = dyn_cast<DeclStmt>(s)) {
            std::string out;
            for (const Decl* d : ds->decls()) {
                if (const auto* vd = dyn_cast<VarDecl>(d)) out += var_decl(vd, depth);
                else if (isa<StaticAssertDecl>(d) || isa<TypedefNameDecl>(d) || isa<UsingDecl>(d) || isa<UsingShadowDecl>(d)) continue;
                else if (isa<CXXRecordDecl>(d)) continue;
            }
            return out;
        }
        if (const auto* rs = dyn_cast<ReturnStmt>(s)) {
            if (!rs->getRetValue()) return with_pending(depth, "return;");
            if (fn->returns_self) return with_pending(depth, "return;");
            if (fn->tail_loop) {
                if (const CallExpr* tc = self_call(rs->getRetValue(), fn->decl)) return with_pending(depth, tail_call(tc));
            }
            if (fn->returns_address) {
                LV lv = lvalue(rs->getRetValue());
                return with_pending(depth, "return " + address(lv, "returned reference") + ";");
            }
            QualType rt = fn->return_type.getNonReferenceType();
            if (rt->isVoidType()) return with_pending(depth, rvalue(rs->getRetValue()) + ";\n" + ind(depth) + "return;");
            std::string v = rvalue_as(rs->getRetValue(), rt);
            return with_pending(depth, "return " + v + ";");
        }
        if (const auto* is = dyn_cast<IfStmt>(s)) {
            if (is->isConstexpr()) {
                std::optional<const Stmt*> taken = is->getNondiscardedCase(Ctx);
                if (taken && *taken) return stmt(*taken, depth);
                return "";
            }
            std::string out;
            if (is->getInit()) out += stmt(is->getInit(), depth);
            if (const VarDecl* cv = is->getConditionVariable()) out += var_decl(cv, depth);
            std::string cond = rvalue_as(is->getCond(), Ctx.BoolTy);
            out += with_pending(depth, "");
            out += ind(depth) + "if (" + cond + ") {\n" + stmt(is->getThen(), depth + 1) + ind(depth) + "}";
            if (is->getElse()) out += " else {\n" + stmt(is->getElse(), depth + 1) + ind(depth) + "}";
            return out + "\n";
        }
        if (const auto* fs = dyn_cast<ForStmt>(s)) {
            // for (init; cond; inc) body  →  { init; while (true) { if (!cond) break; body; inc; } }
            // so that conditions with temporaries and any init work; continue needs the inc, so use a flag loop
            std::string out = ind(depth) + "{\n";
            if (fs->getInit()) out += stmt(fs->getInit(), depth + 1);
            std::string cond = fs->getCond() ? rvalue_as(fs->getCond(), Ctx.BoolTy) : "true";
            std::string cond_pre = with_pending(depth + 2, "");
            std::string inc;
            if (fs->getInc()) {
                std::string e = rvalue(fs->getInc());
                inc = with_pending(depth + 2, e + ";");
            }
            if (cond_pre.empty() && inc.find('\n') == inc.rfind('\n') && !contains_continue(fs->getBody())) {
                std::string incx = inc.empty() ? "" : inc.substr(inc.find_first_not_of(' '));
                while (!incx.empty() && (incx.back() == '\n' || incx.back() == ';')) incx.pop_back();
                out += ind(depth + 1) + "for (; " + cond + "; " + incx + ") {\n" + stmt(fs->getBody(), depth + 2) +
                       ind(depth + 1) + "}\n";
            } else {
                out += ind(depth + 1) + "bool vk_first = true;\n";
                out += ind(depth + 1) + "while (true) {\n";
                out += ind(depth + 2) + "if (!vk_first) {\n" + indent_block(inc, 1) + ind(depth + 2) + "}\n";
                out += ind(depth + 2) + "vk_first = false;\n";
                out += cond_pre + ind(depth + 2) + "if (!(" + cond + ")) break;\n";
                out += stmt(fs->getBody(), depth + 2);
                out += ind(depth + 1) + "}\n";
            }
            return out + ind(depth) + "}\n";
        }
        if (const auto* ws = dyn_cast<WhileStmt>(s)) {
            if (ws->getConditionVariable()) throw Unsupported("while with a declaration");
            std::string cond = rvalue_as(ws->getCond(), Ctx.BoolTy);
            std::string pre = with_pending(depth + 1, "");
            if (pre.empty())
                return ind(depth) + "while (" + cond + ") {\n" + stmt(ws->getBody(), depth + 1) + ind(depth) + "}\n";
            return ind(depth) + "while (true) {\n" + pre + ind(depth + 1) + "if (!(" + cond + ")) break;\n" +
                   stmt(ws->getBody(), depth + 1) + ind(depth) + "}\n";
        }
        if (const auto* ds = dyn_cast<DoStmt>(s)) {
            std::string body = stmt(ds->getBody(), depth + 1);
            std::string cond = rvalue_as(ds->getCond(), Ctx.BoolTy);
            std::string pre = with_pending(depth + 1, "");
            return ind(depth) + "do {\n" + body + pre + ind(depth) + "} while (" + cond + ");\n";
        }
        if (const auto* ss = dyn_cast<SwitchStmt>(s)) {
            std::string cond = rvalue(ss->getCond());
            std::string t = type(ss->getCond()->getType());
            if (t != "int" && t != "uint") cond = "int(" + cond + ")";
            std::string out = with_pending(depth, "");
            return out + ind(depth) + "switch (" + cond + ") {\n" + stmt(ss->getBody(), depth + 1) + ind(depth) + "}\n";
        }
        if (const auto* cs = dyn_cast<CaseStmt>(s)) {
            std::string v;
            if (!constant(cs->getLHS(), v)) throw Unsupported("non-constant case");
            if (v.back() == 'l') v.pop_back();
            if (v.size() > 1 && v.substr(v.size() - 2) == "ul") v = v.substr(0, v.size() - 2) + "u";
            return ind(depth - 1) + "case " + v + ":\n" + stmt(cs->getSubStmt(), depth);
        }
        if (const auto* dfs = dyn_cast<DefaultStmt>(s)) return ind(depth - 1) + "default:\n" + stmt(dfs->getSubStmt(), depth);
        if (isa<BreakStmt>(s)) return ind(depth) + "break;\n";
        if (isa<ContinueStmt>(s)) return ind(depth) + "continue;\n";
        if (isa<NullStmt>(s)) return "";
        if (const auto* e = dyn_cast<Expr>(s)) {
            std::string v = rvalue(e);
            if (v.empty()) return with_pending(depth, "");
            return with_pending(depth, v + ";");
        }
        throw Unsupported(std::string("statement ") + s->getStmtClassName());
    }

    static std::string indent_block(const std::string& s, int n) {
        std::string pad(n * 4, ' ');
        std::string out;
        size_t start = 0;
        while (start < s.size()) {
            size_t end = s.find('\n', start);
            if (end == std::string::npos) end = s.size();
            out += pad + s.substr(start, end - start) + "\n";
            start = end + 1;
        }
        return out;
    }

    static bool contains_continue(const Stmt* s) {
        if (!s) return false;
        if (isa<ContinueStmt>(s)) return true;
        if (isa<ForStmt>(s) || isa<WhileStmt>(s) || isa<DoStmt>(s)) return false;   // belongs to the inner loop
        for (const Stmt* c : s->children()) if (contains_continue(c)) return true;
        return false;
    }

    // ======================================================================
    //  Kernel entry
    // ======================================================================

    std::string entry(const FunctionDecl* kernel) {
        Fn f;
        fn = &f;
        std::string body;
        long off = 0;
        for (unsigned i = 0; i < kernel->getNumParams(); i++) {
            const ParmVarDecl* p = kernel->getParamDecl(i);
            QualType t = p->getType();
            if (t->isReferenceType()) throw Unsupported("reference kernel parameter");
            long a = std::max(1L, align_of(t));
            off = (off + a - 1) / a * a;
            // the argument lives in the argument buffer: memory
            LV lv;
            lv.kind = LV::Mem;
            lv.type = t;
            lv.text = "arg" + std::to_string(i);
            body += "    uint64_t arg" + std::to_string(i) + " = vk_args + " + std::to_string(off) + "ul;\n";
            f.vars[p] = lv;
            off += size_of(t);
            if (bare(t)->isRecordType()) record(bare(t)->getAsRecordDecl());   // layout check
        }
        body += stmt(kernel->getBody(), 1);
        fn = nullptr;
        return "void main() {\n" + body + "}\n";
    }
};

// ---------------------------------------------------------------------------
//  Finding the launched kernels
// ---------------------------------------------------------------------------

struct LaunchedKernel {
    const FunctionDecl* kernel = nullptr;
    std::string key;   // typeid(vulkcc::KernelTag<kernel>).name()
    std::string name;
};

inline const DeclContext* vulkcc_namespace(ASTContext& ctx) {
    for (Decl* d : ctx.getTranslationUnitDecl()->decls()) {
        if (auto* ns = dyn_cast<NamespaceDecl>(d)) {
            if (ns->getName() == "vulkcc") return ns;
        }
    }
    return nullptr;
}

inline std::vector<LaunchedKernel> launched_kernels(ASTContext& ctx) {
    std::vector<LaunchedKernel> out;
    std::unique_ptr<MangleContext> mangler(ItaniumMangleContext::create(ctx, ctx.getDiagnostics()));
    std::set<const FunctionDecl*> seen;
    // every namespace vulkcc block (the namespace can be reopened)
    for (Decl* d : ctx.getTranslationUnitDecl()->decls()) {
        auto* ns = dyn_cast<NamespaceDecl>(d);
        if (!ns || ns->getName() != "vulkcc") continue;
        for (Decl* m : ns->decls()) {
            auto* ctd = dyn_cast<ClassTemplateDecl>(m);
            if (!ctd || ctd->getName() != "KernelTag") continue;
            for (ClassTemplateSpecializationDecl* spec : ctd->specializations()) {
                const TemplateArgumentList& args = spec->getTemplateArgs();
                if (args.size() != 1) continue;
                const TemplateArgument& a = args[0];
                const FunctionDecl* k = nullptr;
                if (a.getKind() == TemplateArgument::Declaration) k = dyn_cast<FunctionDecl>(a.getAsDecl());
                if (!k || !seen.insert(k->getCanonicalDecl()).second) continue;
                if (const FunctionDecl* def = k->getDefinition()) k = def;
                std::string s;
                llvm::raw_string_ostream os(s);
                mangler->mangleCXXRTTIName(ctx.getRecordType(spec), os);
                os.flush();
                if (s.rfind("_ZTS", 0) == 0) s = s.substr(4);
                LaunchedKernel lk;
                lk.kernel = k;
                lk.key = s;
                lk.name = k->getQualifiedNameAsString();
                if (const TemplateArgumentList* ta = k->getTemplateSpecializationArgs()) {
                    std::string targs;
                    llvm::raw_string_ostream to(targs);
                    printTemplateArgumentList(to, ta->asArray(), PrintingPolicy(ctx.getLangOpts()));
                    to.flush();
                    lk.name += targs;
                }
                out.push_back(lk);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  SPIR-V + registration
// ---------------------------------------------------------------------------

inline bool compile_compute(const std::string& glsl, const std::string& tmp_base, std::vector<uint32_t>& words,
                            std::string& log) {
    std::string src = tmp_base + ".comp";
    std::string spv = tmp_base + ".spv";
    {
        std::ofstream f(src);
        f << glsl;
    }
    std::string cmd = "glslangValidator -V --target-env vulkan1.2 -S comp -o " + spv + " " + src + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { log = "could not run glslangValidator"; return false; }
    char buf[256];
    log.clear();
    while (fgets(buf, sizeof buf, pipe)) log += buf;
    int rc = pclose(pipe);
    std::remove(src.c_str());
    if (rc != 0) return false;
    std::ifstream in(spv, std::ios::binary);
    in.seekg(0, std::ios::end);
    size_t n = in.tellg();
    in.seekg(0);
    words.resize(n / 4);
    in.read((char*)words.data(), n);
    std::remove(spv.c_str());
    return true;
}

// Translates every launched kernel; writes C++ registering the SPIR-V to `path`.
inline int write_kernels(ASTContext& ctx, const SourceManager& sm, const std::string& path, bool keep_glsl) {
    std::vector<LaunchedKernel> kernels = launched_kernels(ctx);
    std::string cpp = "// Generated by vulkcc: the Vulkan kernels of this program.\n"
                      "#include \"ops/vulkan/runtime.hpp\"\n\nnamespace {\n\n";
    int written = 0, skipped = 0;
    for (size_t i = 0; i < kernels.size(); i++) {
        const LaunchedKernel& lk = kernels[i];
        std::string glsl;
        try {
            if (!lk.kernel->hasBody()) throw Unsupported("kernel has no body");
            Kernel k(ctx, sm);
            glsl = k.translate(lk.kernel);
        } catch (const Unsupported& e) {
            llvm::errs() << "vulkcc: skipping kernel " << lk.name << ": " << e.what() << "\n";
            skipped++;
            continue;
        }
        if (keep_glsl) std::ofstream(path + "." + std::to_string(i) + ".comp") << "// " << lk.name << "\n" << glsl;
        std::vector<uint32_t> words;
        std::string log;
        if (!compile_compute(glsl, path + ".tmp" + std::to_string(i), words, log)) {
            llvm::errs() << "vulkcc: kernel " << lk.name << " did not compile to SPIR-V:\n" << log;
            if (!keep_glsl) llvm::errs() << "(set VULKCC_KEEP=1 to keep the generated GLSL)\n";
            skipped++;
            continue;
        }
        std::string id = "k" + std::to_string(i);
        cpp += "// " + lk.name + "\nconst uint32_t " + id + "[] = {";
        for (size_t w = 0; w < words.size(); w++) {
            if (w % 8 == 0) cpp += "\n    ";
            char b[16];
            snprintf(b, sizeof b, "0x%08x,", words[w]);
            cpp += b;
        }
        cpp += "\n};\nconst int " + id + "_registered = vulkcc::register_kernel(\"" + lk.key + "\", " + id + ", " +
               std::to_string(words.size()) + ");\n\n";
        written++;
    }
    cpp += "} // namespace\n";
    std::ofstream(path) << cpp;
    llvm::outs() << "vulkcc: " << written << " kernels compiled";
    if (skipped) llvm::outs() << ", " << skipped << " skipped";
    llvm::outs() << "\n";
    return written;
}

// ---------------------------------------------------------------------------
//  Rewriting kernel<<<...>>>(args) → vulkcc::launch<kernel>(..., args)
// ---------------------------------------------------------------------------

class LaunchFinder : public RecursiveASTVisitor<LaunchFinder> {
public:
    std::vector<const CUDAKernelCallExpr*> launches;
    bool shouldVisitTemplateInstantiations() const { return false; }
    bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr* e) {
        launches.push_back(e);
        return true;
    }
};

inline std::string source_text(SourceRange r, const SourceManager& sm, const LangOptions& lo) {
    return Lexer::getSourceText(CharSourceRange::getTokenRange(r), sm, lo).str();
}

// Returns the main file with every launch rewritten; launches outside the
// main file are reported.
inline std::string rewrite_launches(ASTContext& ctx, bool& ok) {
    const SourceManager& sm = ctx.getSourceManager();
    const LangOptions& lo = ctx.getLangOpts();
    LaunchFinder finder;
    finder.TraverseDecl(ctx.getTranslationUnitDecl());
    FileID main = sm.getMainFileID();
    std::string text = sm.getBufferData(main).str();
    struct Edit { unsigned begin, end; std::string with; };
    std::vector<Edit> edits;
    std::set<unsigned> done;
    ok = true;
    for (const CUDAKernelCallExpr* e : finder.launches) {
        SourceLocation b = sm.getExpansionLoc(e->getBeginLoc());
        SourceLocation en = sm.getExpansionLoc(e->getEndLoc());
        if (sm.getFileID(b) != main) {
            llvm::errs() << "vulkcc: " << b.printToString(sm)
                         << ": kernel<<<...>>> launches are only rewritten in the main file; use vulkcc::launch<kernel>(grid, block, shmem, stream, args...)\n";
            ok = false;
            continue;
        }
        unsigned bo = sm.getFileOffset(b);
        if (!done.insert(bo).second) continue;
        const CallExpr* config = e->getConfig();
        std::string callee = source_text(e->getCallee()->getSourceRange(), sm, lo);
        std::vector<std::string> cfg;
        for (unsigned i = 0; i < config->getNumArgs() && i < 4; i++) {
            const Expr* a = config->getArg(i);
            if (isa<CXXDefaultArgExpr>(a)) cfg.push_back(i == 3 ? "nullptr" : "0");
            else cfg.push_back(source_text(a->getSourceRange(), sm, lo));
        }
        while (cfg.size() < 4) cfg.push_back(cfg.size() == 3 ? "nullptr" : "0");
        std::string call = "::vulkcc::launch<" + callee + ">(" + cfg[0] + ", " + cfg[1] + ", " + cfg[2] + ", " + cfg[3];
        for (unsigned i = 0; i < e->getNumArgs(); i++) {
            if (isa<CXXDefaultArgExpr>(e->getArg(i))) continue;
            call += ", " + source_text(e->getArg(i)->getSourceRange(), sm, lo);
        }
        call += ")";
        unsigned eo = sm.getFileOffset(Lexer::getLocForEndOfToken(en, 0, sm, lo));
        edits.push_back({bo, eo, call});
    }
    std::sort(edits.begin(), edits.end(), [](const Edit& a, const Edit& b) { return a.begin > b.begin; });
    for (auto& ed : edits) text.replace(ed.begin, ed.end - ed.begin, ed.with);
    return text;
}

} // namespace vulkcc
