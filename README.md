# TensorFlow-Island
RemObjects Island platform bindings for TensorFlow C API v1.15.0.

TensorFlow-Island is a high-level abstraction of TensorFlow C-APIs, in several modern languages: Swift, Oxygene, Java, Go and C#.  It is dependent on RemObjects Elements LLVM-based Island platform compilers, genenating CPU-native code for machine-learning applications on Windows, Linux, and MacOS.

Note - these supported languages (Swift, Oxygene, Java, Go and C#) are all compiled into CPU native code, without any dependency on .NET CLR, JVM, or any virtual machine environment. This is a unique capability of RemObjects Elements compiler toolchain. This particularly fits the design objective and goal of TenorFlow-Island, hence we select this toolchain.

TensorFlow-Island is initially inspired by TensorFlow4Delphi (https://github.com/hartmutdavid/TensorFlow4Delphi). The framework design is also infuenced by TensorFlowSharp https://github.com/migueldeicaza/TensorFlowSharp,  with our own insights, adjustments, and enhancements.

The following diagram illustrates the TensorFlow-Island architecture.
![TensorFlow-Island Diagram](../master/Images/TensorFlow-Island-Diagram.png) 

# Difference with other TensorFlow bindings
The difference between TensorFlow-Island and other wellknown TensorFlow bindings, for example, TensorFlow.NET (https://github.com/SciSharp/TensorFlow.NET) is:
- Multiple programming language support, including Oxygene, Swift, Java, Go, and C#, thanks to RemObjects LLVM-based Island platform compilers;
- Compiled code runs directly on CPU, without the dependency on JVM, .NET CLR, or Python/CPython intepreter. 
- Direct acces to TensorFlow C API; no .NET P-Invoke involved;
- TensorFlow-Island itself is a light-weigth abstraction. Unlike TensorFlow.NET, TensorFlow-Island does not intend for a direct translation of existing Python-based TensorFlow code. Rather, the design is to have a CPU native bindings of TensorFlow C API with modern language features (e.g., the Dispose Pattern, Lamda Expression, LinQ) to help manage resources, streamline model development, and efficient run time performance, all in one package.

# Design Objectives
 - A higher level abstraction of TensorFlow C-API,  cross-platform (Windows, Linux and MacOS), and CPU/GPU native machine code;
 - Multiple language support for Oxygene, Swift, Java, Go and C#, and multiple platform support for Windows, Linux and MacOS;
 - Support performance-critical machine-learning and Artifical Intelligence algorithms;
 - Provides a foundational Computational Graph framework, with an additional set of customized TensorFlow Ops for Traffic and Transportation applications, including Traffic Signal Optimizations, Smart Driver APIs for Connected Vehicle Application Simulation, and Innovative Traffic Simulation Calibration;
 - Provides a foundational Computational Graph framework to be integrated in PTV Vissim Microscopic Traffic Simulator, enabling GPU computing for External Driver Model, and Signal Control API modules.

# Compilers Toolchain Requirements
TensorFlow Island requires RemObjects Elements Compilers. For commmercial, open-source, or academic applications, please contact RemObjects (https://www.elementscompiler.com/elements/) for different licensing options.

# License
MIT License (c) 2019-2020. Copywright Wuping Xin and KLD Engineering, P. C. 
