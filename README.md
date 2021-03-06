# TensorFlow-Island
RemObjects Island platform bindings for TensorFlow C API v2.3.0. Designed for high-performance AI/ML-Integerated Intelligent Transport Systems (ITS) applications.

# CPU-Native Multi-language Multi-Platform Support
TensorFlow-Island is a high-level abstraction of TensorFlow C-API, in several modern languages: Swift, Oxygene, Java, Go and C#.  Tensor-Island is dependent on the [RemObjects Elements](https://www.remobjects.com) compiler, which is based on the LLVM compiler infrastructure capable of genenating CPU-native machine code on Windows, Linux, and MacOS.

The languages (Swift, Oxygene, Java, Go and C#) as supported by TensorFlow-Island can be mixed interchangably at source code level. They are all compiled into CPU-native machine code by the RemObject Elements compiler, without dependencies on .NET CLR, JVM, or any virtual machine environment. This fits the design objectives (see below) of TenorFlow-Island, hence the reason (for the selection of RemObject Elements compiler).

# Inspiration
TensorFlow-Island is initially inspired by [TensorFlow4Delphi](https://github.com/hartmutdavid/TensorFlow4Delphi). The framework design is also infuenced by [TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp), with new insights, adjustments, and enhancements.

The following diagram illustrates the TensorFlow-Island architecture.

![TensorFlow-Island Diagram](../master/Images/TensorFlow-Island-Diagram.png) 

# Difference with other TensorFlow bindings
The difference between TensorFlow-Island and other wellknown TensorFlow bindings, for example, [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET) is:
- Multiple programming language support, including Oxygene, Swift, Java, Go, and C#, thanks to the RemObjects LLVM-based Elements compiler;
- CPU-native machine code, without dependencies on JVM, .NET CLR, or Python/CPython intepreter; 
- Direct acces to TensorFlow C API; no .NET P-Invoke, marshalling, or JNI wrappers involved;
- TensorFlow-Island itself is a light-weight abstraction. Unlike TensorFlow.NET, TensorFlow-Island does not intend for a direct translation of existing Python-based TensorFlow code. Rather, the design is to have a CPU-native binding of TensorFlow C API with modern language features (e.g., the Dispose Pattern, Lamda Expression, LinQ) to help manage resources, streamline model development, and efficient run-time performance, all in one package.

# Design Objectives
 - A higher level abstraction of TensorFlow C-API,  cross-platform (Windows, Linux and MacOS), and CPU-native machine code;
 - Multiple language support for Oxygene, Swift, Java, Go and C#, and multiple platforms support for Windows, Linux and MacOS;
 - Support performance-critical machine-learning and Artifical Intelligence algorithms;
 - Provides a foundational Computational Graph framework, with an additional set of customized TensorFlow Ops for Traffic and Transportation applications, including Traffic Signal Optimizations, Smart Driver API for Connected Vehicle Application Simulation, and Innovative Traffic Simulation Calibration;
 - Provides a foundational Computational Graph framework to be integrated in PTV Vissim Microscopic Traffic Simulator, enabling GPU computing for External Driver Model, and Signal Control API modules.

# Compilers Toolchain Requirements
TensorFlow Island requires the RemObjects Elements compiler. For commmercial, open-source, or academic applications, please contact [RemObjects](https://www.elementscompiler.com/elements/) for different licensing options.

# License
MIT License (c) 2019-2020. Copyright [Wuping Xin](wupingxin.net) and [KLD Engineering, P. C.](www.kldcompanies.com) 
