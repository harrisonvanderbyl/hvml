#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Lex/Lexer.h>
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
    return type.str();
}

/// ---------------- Shader Visitor
class ShaderVisitor : public RecursiveASTVisitor<ShaderVisitor> {
public:
    ShaderVisitor(ASTContext &ctx) : Ctx(ctx), SM(ctx.getSourceManager()) {}

    bool VisitCXXRecordDecl(CXXRecordDecl *decl) {
        if (!decl->hasAttrs()) return true;
        
        for (auto *attr : decl->attrs()) {
            if (auto *ann = dyn_cast<AnnotateAttr>(attr)) {
                if (ann->getAnnotation() == "shader") {
                    processShaderStruct(decl);
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
    };

    std::vector<GLSLShader> shaders;

private:
    ASTContext &Ctx;
    SourceManager &SM;

    void processShaderStruct(CXXRecordDecl *decl) {
        std::vector<std::pair<std::string, std::string>> uniforms;
        std::vector<std::pair<std::string, std::string>> vertexInputs;
        std::vector<std::pair<std::string, std::string>> fragmentInputs;

        // Collect struct members -> uniforms
        for (auto *field : decl->fields()) {
            std::string type = VertexAttributeToGLType(field->getType().getAsString());
            std::string name = field->getNameAsString();
            uniforms.push_back({type, name});
        }

        // Find vertex() and fragment() methods
        CXXMethodDecl *vertexMethod = nullptr;
        CXXMethodDecl *fragmentMethod = nullptr;

        for (auto *method : decl->methods()) {
            if (!method->getIdentifier()) continue;
            std::string name = method->getNameAsString();
            
            if (name == "vertex") {
                vertexMethod = method;
                // Collect vertex inputs
                for (auto *param : method->parameters()) {
                    std::string type = VertexAttributeToGLType(param->getType().getAsString());
                    std::string pname = param->getNameAsString();
                    vertexInputs.push_back({type, pname});
                }
            } else if (name == "fragment") {
                fragmentMethod = method;
                // Collect fragment inputs (these become varyings)
                for (auto *param : method->parameters()) {
                    std::string type = VertexAttributeToGLType(param->getType().getAsString());
                    std::string pname = param->getNameAsString();
                    fragmentInputs.push_back({type, pname});
                }
            }
        }

        if (vertexMethod && fragmentMethod) {
            std::string vertexBody = getSourceText(vertexMethod->getBody());
            std::string fragmentBody = getSourceText(fragmentMethod->getBody());
            
            shaders.push_back(generateGLSL(uniforms, vertexInputs, fragmentInputs, 
                                          vertexBody, fragmentBody, 
                                            SM.getFilename(decl->getLocation()).str(), decl->getNameAsString()));
        }
    }

    std::string convertFragmentBody(const std::string &body) {
        std::string out = body;
        // Types
        out = std::regex_replace(out, std::regex("\\bfloat32x2\\b"), "vec2");
        out = std::regex_replace(out, std::regex("\\bfloat32x3\\b"), "vec3");
        out = std::regex_replace(out, std::regex("\\bfloat32x4\\b"), "vec4");
        out = std::regex_replace(out, std::regex("\\buint84\\b"), "vec4");
        
        // discard() -> discard
        out = std::regex_replace(out, std::regex("discard\\(\\);"), "discard;");
        
        // Handle .xyz() -> .xyz (remove parentheses from all swizzle operators)
        for (const std::string& swizzlea : {"x", "y", "z", "w"}) {
            for (const std::string& swizzleb : {"x", "y", "z", "w", ""}) {
                for (const std::string& swizzlec : {"x", "y", "z", "w", ""}) {
                    for (const std::string& swizzled : {"x", "y", "z", "w", ""}) {
                        std::string swizzle = swizzlea + swizzleb + swizzlec + swizzled;
                        std::string pattern = "\\." + swizzle + "\\(\\)";
                        std::string replacement = "." + swizzle;
                        out = std::regex_replace(out, std::regex(pattern), replacement);
                    }
                }
            }
        }
        return out;
    }

    std::string convertVertexBody(const std::string &body, 
                                  const std::vector<std::pair<std::string, std::string>>& fragmentInputs,
                                  const std::vector<std::pair<std::string, std::string>>& vertexInputs) {
        std::string out = body;
        
        // Replace vertex parameter names with 'a' prefixed names (aPos, aColor, aDensity)
        for (auto &vi : vertexInputs) {
            std::string oldName = "\\b" + vi.second + "\\b";
            std::string newName = "a" + capitalize(vi.second);
            out = std::regex_replace(out, std::regex(oldName), newName);
        }
        
        // Types
        out = std::regex_replace(out, std::regex("\\bfloat32x2\\b"), "vec2");
        out = std::regex_replace(out, std::regex("\\bfloat32x3\\b"), "vec3");
        out = std::regex_replace(out, std::regex("\\bfloat32x4\\b"), "vec4");
        out = std::regex_replace(out, std::regex("\\buint84\\b"), "vec4");
        
        // Remove parentheses from swizzle operators
        out = std::regex_replace(out, std::regex("\\.xyz\\(\\)"), ".xyz");
        out = std::regex_replace(out, std::regex("\\.xy\\(\\)"), ".xy");
        out = std::regex_replace(out, std::regex("\\.xyzw\\(\\)"), ".xyzw");
        out = std::regex_replace(out, std::regex("\\.w\\(\\)"), ".w");
        out = std::regex_replace(out, std::regex("\\.x\\(\\)"), ".x");
        out = std::regex_replace(out, std::regex("\\.y\\(\\)"), ".y");
        out = std::regex_replace(out, std::regex("\\.z\\(\\)"), ".z");
        
        return out;
    }
    
    std::string generateFragmentHelperFunction(
        const std::vector<std::pair<std::string, std::string>>& fragmentInputs) {
        std::string func = "void fragment(";
        
        // Parameters
        for (size_t i = 0; i < fragmentInputs.size(); i++) {
            func += fragmentInputs[i].first + " in_" + fragmentInputs[i].second;
            if (i < fragmentInputs.size() - 1) func += ", ";
        }
        func += ") {\n";
        
        // Assignments
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

    std::string getSourceText(Stmt *S) {
        if (!S) return "";
        CharSourceRange CR = CharSourceRange::getTokenRange(S->getSourceRange());
        return Lexer::getSourceText(CR, SM, Ctx.getLangOpts()).str();
    }

    GLSLShader generateGLSL(
        const std::vector<std::pair<std::string, std::string>>& uniforms,
        const std::vector<std::pair<std::string, std::string>>& vertexInputs,
        const std::vector<std::pair<std::string, std::string>>& fragmentInputs,
        const std::string &vertexBody,
        const std::string &fragmentBody,
        const std::string &original_filename = "",
        const std::string &struct_name = ""
    ) {
        GLSLShader s;

        // ---- Vertex Shader
        std::string vs;
        vs += "#version 330 core\n";
        
        // Input attributes
        for (size_t i = 0; i < vertexInputs.size(); i++)
            vs += "layout (location = " + std::to_string(i) + ") in " + 
                  vertexInputs[i].first + " a" + 
                  capitalize(vertexInputs[i].second) + ";\n";
        
        // Uniforms
        for (auto &u : uniforms)
            vs += "uniform " + u.first + " " + u.second + ";\n";
        
        // Varyings (outputs) - use fragment parameter names directly
        for (auto &fi : fragmentInputs)
            vs += "out " + fi.first + " " + fi.second + ";\n";
        
        vs += "\n";
        
        // Add fragment helper function
        vs += generateFragmentHelperFunction(fragmentInputs);
        
        vs += "void main() {\n";
        vs += convertVertexBody(vertexBody, fragmentInputs, vertexInputs);
        vs += "\n}\n";
        
        s.vertex = vs;

        // ---- Fragment Shader
        std::string fs;
        fs += "#version 330 core\n";
        fs += "out vec4 FragColor;\n";
        
        // Varyings (inputs) - use fragment parameter names directly
        for (auto &fi : fragmentInputs)
            fs += "in " + fi.first + " " + fi.second + ";\n";
        
        // Uniforms (only if used in fragment shader)
        for (auto &u : uniforms)
            fs += "uniform " + u.first + " " + u.second + ";\n";
        
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
    explicit ShaderASTConsumer(ASTContext &ctx, std::string outputPath, std::string inputPath = "") 
        : Visitor(ctx), OutputPath(outputPath), InputPath(inputPath) {}

    void HandleTranslationUnit(ASTContext &ctx) override {
        Visitor.TraverseDecl(ctx.getTranslationUnitDecl());
        std::string headerOutput;
        headerOutput += "#include \"display/materials/materials.hpp\"\n";
        for (auto &s : Visitor.shaders) {
            llvm::outs() << "==== Vertex Shader ====\n" << s.vertex << "\n";
            llvm::outs() << "==== Fragment Shader ====\n" << s.fragment << "\n";
            
            // Compile shaders to SPIR-V binary
            headerOutput+=compileAndGenerateHeader(s);
        }

        headerOutput += "#include \"" + this->InputPath + "\"\n\n"; 
        // Write to output header file
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
    std::string OutputPath;
    std::string InputPath;
    
    bool compileGLSLToSPIRV(const std::string& glslSource, 
                            const std::string& shaderType,
                            std::vector<uint8_t>& spirvBinary) {
        // Write GLSL to temporary file
        std::string tempGLSL = "/tmp/shader_" + shaderType + ".glsl";
        std::string tempSPV = "/tmp/shader_" + shaderType + ".spv";
        
        std::ofstream glslFile(tempGLSL);
        if (!glslFile) {
            llvm::errs() << "Failed to create temporary GLSL file\n";
            return false;
        }
        glslFile << glslSource;
        glslFile.close();
        
        // Compile with glslangValidator
        std::string cmd = "glslangValidator --auto-map-locations -G -S " + shaderType + 
                         " -o " + tempSPV + " " + tempGLSL + " 2>&1";
        
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            llvm::errs() << "Failed to run glslangValidator\n";
            return false;
        }
        
        char buffer[128];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        int returnCode = pclose(pipe);
        
        if (returnCode != 0) {
            llvm::errs() << "Shader compilation failed:\n" << result << "\n";
            return false;
        }
        
        // Read SPIR-V binary
        std::ifstream spvFile(tempSPV, std::ios::binary);
        if (!spvFile) {
            llvm::errs() << "Failed to read SPIR-V binary\n";
            return false;
        }
        
        spvFile.seekg(0, std::ios::end);
        size_t size = spvFile.tellg();
        spvFile.seekg(0, std::ios::beg);
        
        spirvBinary.resize(size);
        spvFile.read(reinterpret_cast<char*>(spirvBinary.data()), size);
        spvFile.close();
        
        // Cleanup
        std::remove(tempGLSL.c_str());
        std::remove(tempSPV.c_str());
        
        return true;
    }
    
    std::string compileAndGenerateHeader(const ShaderVisitor::GLSLShader& shader) {
        
        llvm::outs() << "\n==== Compiling Shaders to SPIR-V ====\n";
        
        std::string output = "";
        
        output += "template <>\nstruct Shader<";
        output += shader.struct_name + "> : public Material {\n";
        
        // Vertex shader binary
        output += "const char* getFragmentShaderSource() override {\n    ";
        output += "return R\"(";;
        output += shader.fragment;
        output += ")\";\n";
        output += "} \n\n";
        
        // Fragment shader binary
        output += "const char* getVertexShaderSource() override {\n    ";
        output += "return R\"(";;
        output += shader.vertex;
        output += ")\";\n";
        output += "} \n\n";
        
        output += "} ;\n";

        return output;
    }
};

std::string inputFileName;
/// ---------------- Frontend Action
class ShaderFrontendAction : public ASTFrontendAction {
public:


    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        return std::make_unique<ShaderASTConsumer>(CI.getASTContext(), OutputPath, inputFileName);
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