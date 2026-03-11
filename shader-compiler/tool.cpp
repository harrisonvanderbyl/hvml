#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Lex/Lexer.h>
#include <clang/Sema/Sema.h>
#include <llvm/Support/CommandLine.h>
#include <regex>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory ShaderToolCategory("shader-tool");
static llvm::cl::opt<std::string> OutputPath(
    "o",
    llvm::cl::desc("Output header file path"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("shad.cpp"),
    llvm::cl::cat(ShaderToolCategory)
);

/// --------------------------------------------
/// Your semantic mapping hook
/// --------------------------------------------
static std::string VertexAttributeToGLType(StringRef type) {
    if (type == "float32x2") return "vec2";
    if (type == "float32x3") return "vec3";
    if (type == "float32x4") return "vec4";
    if (type == "uint84") return "vec4";
    if (type == "float") return "float";
    if (type == "uint16_t") return "uint16_t";
    if (type == "uint32_t") return "uint";
    // Sampler types — match exact name, struct-prefixed, and any substring form
    if (type.contains("samplerBuffer")) return "samplerBuffer";
    if (type.contains("sampler2D"))     return "sampler2D";
    // int vector types
    if (type == "int32x2") return "ivec2";
    if (type == "int32x3") return "ivec3";
    if (type == "int32x4") return "ivec4";
    // Canonical C++ types that printPretty may emit for fields
    if (type == "int" || type == "int32_t") return "int";
    if (type == "glm::vec2") return "vec2";
    if (type == "glm::vec3") return "vec3";
    if (type == "glm::vec4") return "vec4";
    if (type == "glm::mat4") return "mat4";
    if (type == "glm::mat3") return "mat3";
    if (type == "glm::ivec2") return "ivec2";
    return type.str();
}

/// --------------------------------------------
/// Pre-pass visitor: collects (ClassTemplateDecl*, TemplateArgument[])
/// for every template specialization that appears in a use context
/// (new-expressions, variable declarations, function call args, etc.)
/// --------------------------------------------
class TemplateUsageCollector : public RecursiveASTVisitor<TemplateUsageCollector> {
public:
    struct Usage {
        ClassTemplateDecl *CTD;
        std::vector<TemplateArgument> Args;
    };

    explicit TemplateUsageCollector(ASTContext &Ctx,
                                    const llvm::DenseSet<ClassTemplateDecl*> &shaderTemplates)
        : Ctx(Ctx), ShaderTemplates(shaderTemplates) {}

    // Catch every specialization type that appears anywhere in the AST
    bool VisitTemplateSpecializationType(TemplateSpecializationType *TST) {
        collectFromTST(TST);
        return true;
    }

    // Also catch record types that are already fully resolved specializations
    bool VisitRecordType(RecordType *RT) {
        auto *spec = dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl());
        if (!spec) return true;
        auto *ctd = spec->getSpecializedTemplate();
        if (!ShaderTemplates.count(ctd)) return true;
        addUsage(ctd, spec->getTemplateArgs().asArray());
        return true;
    }

    const std::vector<Usage> &getUsages() const { return Usages; }

private:
    ASTContext &Ctx;
    const llvm::DenseSet<ClassTemplateDecl*> &ShaderTemplates;
    std::vector<Usage> Usages;
    // Deduplicate by (CTD, serialised arg string)
    llvm::DenseMap<ClassTemplateDecl*, llvm::StringSet<>> Seen;

    void collectFromTST(TemplateSpecializationType *TST) {
        auto TN = TST->getTemplateName();
        auto *ctd = dyn_cast_or_null<ClassTemplateDecl>(TN.getAsTemplateDecl());
        if (!ctd || !ShaderTemplates.count(ctd)) return;
        addUsage(ctd, TST->template_arguments());
    }

    void addUsage(ClassTemplateDecl *ctd, ArrayRef<TemplateArgument> args) {
        // Build a string key for deduplication
        std::string key;
        llvm::raw_string_ostream KS(key);
        for (auto &a : args) {
            a.print(PrintingPolicy(Ctx.getLangOpts()), KS, /*IncludeType=*/true);
            KS << "|";
        }
        KS.flush();
        if (!Seen[ctd].insert(key).second) return; // already seen
        Usages.push_back({ctd, {args.begin(), args.end()}});
    }
};

/// ---------------- Shader Visitor
class ShaderVisitor : public RecursiveASTVisitor<ShaderVisitor> {
public:
    ShaderVisitor(ASTContext &ctx, Sema &sema)
        : Ctx(ctx), SM(ctx.getSourceManager()), SemaRef(sema) {}

    // Handle non-template shader structs — skip specializations (handled below)
    bool VisitCXXRecordDecl(CXXRecordDecl *decl) {
        if (isa<ClassTemplateSpecializationDecl>(decl)) return true;
        if (decl->getDescribedClassTemplate() != nullptr) return true;
        if (!decl->hasAttrs()) return true;

        for (auto *attr : decl->attrs()) {
            if (auto *ann = dyn_cast<AnnotateAttr>(attr)) {
                if (ann->getAnnotation() == "shader") {
                    processShaderStruct(decl, decl->getQualifiedNameAsString());
                }
            }
        }
        return true;
    }

    // Handle template specializations
    bool VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl *decl) {
        if (!decl->isCompleteDefinition()) return true;

        auto *tmpl = decl->getSpecializedTemplate()->getTemplatedDecl();
        if (!tmpl->hasAttrs()) return true;

        for (auto *attr : tmpl->attrs()) {
            if (auto *ann = dyn_cast<AnnotateAttr>(attr)) {
                if (ann->getAnnotation() == "shader") {
                    PrintingPolicy PP(Ctx.getLangOpts());
                    PP.SuppressTagKeyword = true;
                    PP.FullyQualifiedName = false;
                    std::string fullName;
                    llvm::raw_string_ostream OS(fullName);
                    decl->getNameForDiagnostic(OS, PP, /*Qualified=*/false);
                    OS.flush();

                    if (decl->getSpecializationKind() == TSK_ImplicitInstantiation ||
                        decl->getSpecializationKind() == TSK_ExplicitInstantiationDeclaration) {
                        SemaRef.InstantiateClassTemplateSpecializationMembers(
                            decl->getLocation(), decl, TSK_ExplicitInstantiationDefinition);
                    }

                    forceInstantiateMethods(decl);
                    processShaderStruct(decl, fullName);
                }
            }
        }
        return true;
    }

    struct GLSLShader {
        std::string vertex;
        std::string fragment;
        std::string original_filename;
        std::string struct_name;
        std::vector<std::pair<std::string, std::string>> uniforms;
    };

    std::vector<GLSLShader> shaders;

    Sema &getSema() { return SemaRef; }

private:
    ASTContext &Ctx;
    SourceManager &SM;
    Sema &SemaRef;

    void forceInstantiateMethods(CXXRecordDecl *decl) {
        for (auto *d : decl->decls()) {
            if (auto *ftd = dyn_cast<FunctionTemplateDecl>(d)) {
                for (auto *spec : ftd->specializations()) {
                    auto *md = dyn_cast<CXXMethodDecl>(spec);
                    if (!md || md->isInvalidDecl()) continue;
                    if (!md->hasBody()) {
                        SemaRef.InstantiateFunctionDefinition(
                            decl->getLocation(), md,
                            /*Recursive=*/true,
                            /*DefinitionRequired=*/true,
                            /*AtEndOfTU=*/true);
                    }
                }
            }
        }
    }

    void processShaderStruct(CXXRecordDecl *decl, const std::string &structName) {
        std::vector<std::pair<std::string, std::string>> uniforms;
        std::vector<std::pair<std::string, std::string>> vertexInputs;
        std::vector<std::pair<std::string, std::string>> fragmentInputs;

        PrintingPolicy FieldPP(Ctx.getLangOpts());
        FieldPP.SuppressTagKeyword = true;
        FieldPP.FullyQualifiedName = false;
        FieldPP.SuppressScope = true;

        auto isKnownGLSLType = [](const std::string& t) -> bool {
            static const std::vector<std::string> known = {
                "vec2","vec3","vec4","mat3","mat4",
                "ivec2","ivec3","ivec4",
                "float","int","uint","bool",
                "sampler2D","samplerBuffer",
                "uint16_t"
            };
            for (auto &k : known) if (t == k) return true;
            if (t.find('[') != std::string::npos) return true;
            return false;
        };

        for (auto *field : decl->fields()) {
            QualType fieldTy = field->getType();

            std::string typeName = fieldTy.getDesugaredType(Ctx).getAsString(FieldPP);
            std::string glslType = VertexAttributeToGLType(typeName);

            if (!isKnownGLSLType(glslType)) {
                typeName = fieldTy.getCanonicalType().getAsString(FieldPP);
                glslType = VertexAttributeToGLType(typeName);
            }

            if (!isKnownGLSLType(glslType)) {
                typeName = fieldTy.getAsString(FieldPP);
                glslType = VertexAttributeToGLType(typeName);
            }

            std::string name = field->getNameAsString();
            uniforms.push_back({glslType, name});
        }

        CXXMethodDecl *vertexMethod = nullptr;
        CXXMethodDecl *fragmentMethod = nullptr;
        bool hasuint16_t = false;

        for (auto *method : decl->methods()) {
            if (!method->getIdentifier()) continue;
            std::string name = method->getNameAsString();

            if (name == "vertex") {
                vertexMethod = method;
                for (auto *param : method->parameters()) {
                    std::string type = VertexAttributeToGLType(param->getType().getAsString());
                    std::string pname = param->getNameAsString();
                    vertexInputs.push_back({type, pname});
                    if (type == "uint16_t") hasuint16_t = true;
                }
            } else if (name == "fragment" && !fragmentMethod) {
                fragmentMethod = method;
                for (auto *param : method->parameters()) {
                    std::string type = VertexAttributeToGLType(param->getType().getAsString());
                    std::string pname = param->getNameAsString();
                    fragmentInputs.push_back({type, pname});
                }
            }
        }

        if (!fragmentMethod) {
            for (auto *d : decl->decls()) {
                auto *ftd = dyn_cast<FunctionTemplateDecl>(d);
                if (!ftd || ftd->getName() != "fragment") continue;

                for (auto *spec : ftd->specializations()) {
                    auto *md = dyn_cast<CXXMethodDecl>(spec);
                    if (!md || md->isInvalidDecl()) continue;
                    if (!md->hasBody()) continue;

                    fragmentMethod = md;
                    if (fragmentInputs.empty()) {
                        for (auto *param : md->parameters()) {
                            std::string type = VertexAttributeToGLType(param->getType().getAsString());
                            std::string pname = param->getNameAsString();
                            fragmentInputs.push_back({type, pname});
                        }
                    }
                    break;
                }

                if (fragmentMethod) break;
            }
        }

        if (vertexMethod && fragmentMethod) {
            std::string vertexBody = getInstantiatedSourceText(vertexMethod->getBody());
            std::string fragmentBody = getInstantiatedSourceText(fragmentMethod->getBody());

            shaders.push_back(generateGLSL(uniforms, vertexInputs, fragmentInputs,
                                           vertexBody, fragmentBody,
                                           SM.getFilename(decl->getLocation()).str(),
                                           structName, hasuint16_t));
        } else {
            if (!vertexMethod)   llvm::errs() << "WARNING: no vertex() found in " << structName << "\n";
            if (!fragmentMethod) llvm::errs() << "WARNING: no fragment() found in " << structName << "\n";
        }
    }

    std::string convertBody(const std::string &body) {
        std::string out = body;

        out = std::regex_replace(out, std::regex("\\bfloat32x2\\b"), "vec2");
        out = std::regex_replace(out, std::regex("\\bfloat32x3\\b"), "vec3");
        out = std::regex_replace(out, std::regex("\\bfloat32x4\\b"), "vec4");
        out = std::regex_replace(out, std::regex("\\buint84\\b"), "vec4");
        out = std::regex_replace(out, std::regex("\\bint32x2\\b"), "ivec2");
        out = std::regex_replace(out, std::regex("\\bint32x3\\b"), "ivec3");
        out = std::regex_replace(out, std::regex("\\bint32x4\\b"), "ivec4");
        out = std::regex_replace(out, std::regex("\\bint32_t\\b"), "int");
        out = std::regex_replace(out, std::regex("\\buint32_t\\b"), "uint");

        out = std::regex_replace(out, std::regex("discard\\(\\);"), "discard;");

        for (const std::string& a : {"x", "y", "z", "w", "r", "g", "b", "a"}) {
            for (const std::string& b : {"x", "y", "z", "w", "r", "g", "b", "a", ""}) {
                for (const std::string& c : {"x", "y", "z", "w", "r", "g", "b", "a", ""}) {
                    for (const std::string& d : {"x", "y", "z", "w", "r", "g", "b", "a", ""}) {
                        std::string swizzle = a + b + c + d;
                        out = std::regex_replace(out, std::regex("\\." + swizzle + "\\(\\)"), "." + swizzle);
                    }
                }
            }
        }

        out = std::regex_replace(out, std::regex("\\bthis->"), "");
        out = std::regex_replace(out, std::regex("\\bconstexpr\\b\\s*"), "");

        return out;
    }

    std::string convertFragmentBody(const std::string &body) {
        return convertBody(body);
    }

    std::string convertVertexBody(const std::string &body,
                                  const std::vector<std::pair<std::string, std::string>>& fragmentInputs,
                                  const std::vector<std::pair<std::string, std::string>>& vertexInputs) {
        std::string out = convertBody(body);
        for (auto &vi : vertexInputs) {
            out = std::regex_replace(out, std::regex("\\b" + vi.second + "\\b"), "a" + capitalize(vi.second));
        }
        return out;
    }

    std::string generateFragmentHelperFunction(
        const std::vector<std::pair<std::string, std::string>>& fragmentInputs) {
        std::string func = "void fragment(";
        for (size_t i = 0; i < fragmentInputs.size(); i++) {
            func += fragmentInputs[i].first + " in_" + fragmentInputs[i].second;
            if (i < fragmentInputs.size() - 1) func += ", ";
        }
        func += ") {\n";
        for (auto &fi : fragmentInputs) {
            func += "    " + fi.second + " = in_" + fi.second + ";\n";
        }
        func += "}\n\n";
        return func;
    }

    std::string capitalize(const std::string& str) {
        if (str.empty()) return str;
        std::string result = str;
        result[0] = std::toupper(result[0]);
        return result;
    }

    std::string getInstantiatedSourceText(Stmt *S) {
        if (!S) return "";
        std::string result;
        llvm::raw_string_ostream OS(result);
        PrintingPolicy PP(Ctx.getLangOpts());
        PP.SuppressTagKeyword = true;
        PP.FullyQualifiedName = false;
        PP.SuppressScope = true;
        S->printPretty(OS, nullptr, PP);
        OS.flush();
        return result;
    }

    GLSLShader generateGLSL(
        const std::vector<std::pair<std::string, std::string>>& uniforms,
        const std::vector<std::pair<std::string, std::string>>& vertexInputs,
        const std::vector<std::pair<std::string, std::string>>& fragmentInputs,
        const std::string &vertexBody,
        const std::string &fragmentBody,
        const std::string &original_filename = "",
        const std::string &struct_name = "",
        bool hasuint16_t = false
    ) {
        GLSLShader s;

        std::string vs;
        vs += "#version 330 core\n";
        if (hasuint16_t) {
            vs += "#extension GL_NV_gpu_shader5 : enable\n";
        }

        for (size_t i = 0; i < vertexInputs.size(); i++)
            vs += "layout (location = " + std::to_string(i) + ") in " +
                  vertexInputs[i].first + " a" +
                  capitalize(vertexInputs[i].second) + ";\n";

        for (auto &u : uniforms)
            vs += "uniform " + u.first + " " + u.second + ";\n";

        for (auto &fi : fragmentInputs)
            vs += "out " + fi.first + " " + fi.second + ";\n";

        vs += "\n";
        vs += generateFragmentHelperFunction(fragmentInputs);
        vs += "void main() {\n";
        vs += convertVertexBody(vertexBody, fragmentInputs, vertexInputs);
        vs += "\n}\n";

        s.vertex = vs;

        std::string fs;
        fs += "#version 330 core\n";
        fs += "out vec4 FragColor;\n";

        for (auto &fi : fragmentInputs)
            fs += "in " + fi.first + " " + fi.second + ";\n";

        for (auto &u : uniforms)
        {
            fs += "uniform " + u.first + " " + u.second + ";\n";
            s.uniforms.push_back(u);
        }

        fs += "\nvoid main() {\n";
        fs += convertFragmentBody(fragmentBody);
        fs += "\n}\n";

        s.fragment = fs;
        s.original_filename = original_filename;
        s.struct_name = struct_name;

        return s;
    }
};

/// ---------------- AST Consumer
class ShaderASTConsumer : public ASTConsumer {
public:
    explicit ShaderASTConsumer(ASTContext &ctx, Sema &sema,
                               std::string outputPath, std::string inputPath = "")
        : Visitor(ctx, sema), Ctx(ctx), OutputPath(outputPath), InputPath(inputPath) {}

    void HandleTranslationUnit(ASTContext &ctx) override {
        // ----------------------------------------------------------------
        // Step 1: Collect all ClassTemplateDecls annotated with "shader"
        // ----------------------------------------------------------------
        llvm::DenseSet<ClassTemplateDecl*> shaderTemplates;
        for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
            auto *ctd = dyn_cast<ClassTemplateDecl>(decl);
            if (!ctd) continue;
            auto *templated = ctd->getTemplatedDecl();
            if (!templated->hasAttrs()) continue;
            for (auto *attr : templated->attrs()) {
                if (auto *ann = dyn_cast<AnnotateAttr>(attr))
                    if (ann->getAnnotation() == "shader") {
                        shaderTemplates.insert(ctd);
                        break;
                    }
            }
        }

        // ----------------------------------------------------------------
        // Step 2: Walk the entire TU collecting every use of those templates
        // ----------------------------------------------------------------
        TemplateUsageCollector usageCollector(ctx, shaderTemplates);
        usageCollector.TraverseDecl(ctx.getTranslationUnitDecl());

        // ----------------------------------------------------------------
        // Step 3: For each discovered usage, force a complete instantiation
        // ----------------------------------------------------------------
        for (auto &usage : usageCollector.getUsages()) {
            forceInstantiateTemplateWithArgs(
                usage.CTD, usage.Args, Visitor.getSema(), ctx);
        }

        // ----------------------------------------------------------------
        // Step 4: Also force-complete any specializations that already exist
        //         (explicit instantiations, partial specs, etc.)
        // ----------------------------------------------------------------
        for (auto *ctd : shaderTemplates) {
            for (auto *spec : ctd->specializations()) {
                if (!spec->isCompleteDefinition()) {
                    Visitor.getSema().InstantiateClassTemplateSpecializationMembers(
                        spec->getLocation(), spec,
                        TSK_ExplicitInstantiationDefinition);
                }
            }
        }

        // ----------------------------------------------------------------
        // Step 5: Normal traversal — VisitClassTemplateSpecializationDecl
        //         will now find all the specializations we just created
        // ----------------------------------------------------------------
        Visitor.TraverseDecl(ctx.getTranslationUnitDecl());

        // ----------------------------------------------------------------
        // Step 6: Emit output
        // ----------------------------------------------------------------
        std::string headerOutput;
        headerOutput += "#include \"display/materials/materials.hpp\"\n";

        for (auto &s : Visitor.shaders) {
            llvm::outs() << "==== Vertex Shader (" << s.struct_name << ") ====\n" << s.vertex << "\n";
            llvm::outs() << "==== Fragment Shader (" << s.struct_name << ") ====\n" << s.fragment << "\n";
            headerOutput += compileAndGenerateHeader(s);
        }

        headerOutput += "#include \"" + this->InputPath + "\"\n\n";

        std::ofstream outFile(OutputPath);
        if (outFile.is_open()) {
            outFile << headerOutput;
            outFile.close();
            llvm::outs() << "Shader header written to " << OutputPath << "\n";
        } else {
            llvm::errs() << "Failed to write shader header to " << OutputPath << "\n";
        }
    }

private:
    ShaderVisitor Visitor;
    ASTContext &Ctx;
    std::string OutputPath;
    std::string InputPath;

    /// Force a complete instantiation of a class template for the given args,
    /// driven by actual usage found in the TU.
    /// Uses RequireCompleteType so Sema handles all internal bookkeeping safely.
    void forceInstantiateTemplateWithArgs(
        ClassTemplateDecl *ctd,
        ArrayRef<TemplateArgument> args,
        Sema &S,
        ASTContext &ASTCtx)
    {
        // args = the original (possibly sugared) args from the usage site.
        // canonArgs = fully canonical form, used for findSpecialization lookup.
        SmallVector<TemplateArgument, 4> specifiedArgs(args.begin(), args.end());
        SmallVector<TemplateArgument, 4> canonArgs;
        for (auto &arg : args)
            canonArgs.push_back(ASTCtx.getCanonicalTemplateArgument(arg));

        // Check if a complete specialization already exists
        void *insertPos = nullptr;
        auto *existing = ctd->findSpecialization(canonArgs, insertPos);
        if (existing && existing->isCompleteDefinition()) return;

        // getTemplateSpecializationType(TN, specifiedArgs, canonArgs):
        //   - specifiedArgs: the args as written at the usage site (may be sugared)
        //   - canonArgs:     the canonical args used to form the canonical QualType
        TemplateName TN(ctd);
        QualType specTy = ASTCtx.getTemplateSpecializationType(
            TN, specifiedArgs, canonArgs);

        // RequireCompleteType triggers full instantiation via normal Sema machinery.
        SourceLocation loc = ctd->getLocation();
        S.RequireCompleteType(loc, specTy, 0);

        // Find the specialization Sema just created/completed
        auto *spec = ctd->findSpecialization(canonArgs, insertPos);
        if (!spec) {
            llvm::errs() << "WARNING: could not find specialization of "
                         << ctd->getNameAsString() << " after RequireCompleteType\n";
            return;
        }

        // Force all member function bodies to be instantiated too
        if (spec->isCompleteDefinition()) {
            S.InstantiateClassTemplateSpecializationMembers(
                loc, spec, TSK_ExplicitInstantiationDefinition);
        }
    }

    bool compileGLSLToSPIRV(const std::string& glslSource,
                            const std::string& shaderType,
                            std::vector<uint8_t>& spirvBinary) {
        std::string tempGLSL = "/tmp/shader_" + shaderType + ".glsl";
        std::string tempSPV  = "/tmp/shader_" + shaderType + ".spv";

        std::ofstream glslFile(tempGLSL);
        if (!glslFile) { llvm::errs() << "Failed to create temporary GLSL file\n"; return false; }
        glslFile << glslSource;
        glslFile.close();

        std::string cmd = "glslangValidator --auto-map-locations -G -S " + shaderType +
                         " -o " + tempSPV + " " + tempGLSL + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) { llvm::errs() << "Failed to run glslangValidator\n"; return false; }

        char buffer[128];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) result += buffer;
        int returnCode = pclose(pipe);

        if (returnCode != 0) {
            llvm::errs() << "Shader compilation failed:\n" << result << "\n";
            return false;
        }

        std::ifstream spvFile(tempSPV, std::ios::binary);
        if (!spvFile) { llvm::errs() << "Failed to read SPIR-V binary\n"; return false; }

        spvFile.seekg(0, std::ios::end);
        size_t size = spvFile.tellg();
        spvFile.seekg(0, std::ios::beg);
        spirvBinary.resize(size);
        spvFile.read(reinterpret_cast<char*>(spirvBinary.data()), size);
        spvFile.close();

        std::remove(tempGLSL.c_str());
        std::remove(tempSPV.c_str());
        return true;
    }

    std::string compileAndGenerateHeader(const ShaderVisitor::GLSLShader& shader) {
        llvm::outs() << "\n==== Generating Header for " << shader.struct_name << " ====\n";

        std::string output;
        output += "template <>\nstruct Shader<";
        output += shader.struct_name + "> : public Material {\n";

        output += "const char* getFragmentShaderSource() override {\n    ";
        output += "return R\"(";
        output += shader.fragment;
        output += ")\";\n}\n\n";

        output += "const char* getVertexShaderSource() override {\n    ";
        output += "return R\"(";
        output += shader.vertex;
        output += ")\";\n}\n\n";

        output += "void init_uniforms(GLuint shader_program) override {\n    ";
        for (const auto& uniform_pair : shader.uniforms) {
            output += "uniform_setters[\"" + uniform_pair.second + "\"] = UniformSetter(\"" + uniform_pair.second + "\", shader_program);\n    ";
        }
        output += "};\n\n";
        output += "};\n\n";
        return output;
    }
};

std::string inputFileName;

/// ---------------- Frontend Action
class ShaderFrontendAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        return std::make_unique<ShaderASTConsumer>(
            CI.getASTContext(), CI.getSema(), OutputPath, inputFileName);
    }
};

/// ---------------- main
int main(int argc, const char **argv) {
    inputFileName = argv[1];
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ShaderToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    return Tool.run(newFrontendActionFactory<ShaderFrontendAction>().get());
}