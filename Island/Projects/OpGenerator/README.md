#OpGenerator

This directory contains the operations generator, a tool that auto-generate C# code for defining tensorflow ops from the meta data in .proto files.  The generated ops are supplied as a partial TensorFlow.Island.Graph class.

OpGenerator can be used to keep TensorFlow.Island up to date with tensorflow C shared library.

#Usage
- Copy the .proto files in `$(TF_PATH)/tensorflow/core/framework/` to `./ProtoBufs/rx.xx.x/`, where x.xx.x refers to the tensorflow release version.
- Update the copied .proto files, removing those relative paths in the `import` parts.
- Update ProtoGen.bat to set correct input/output paths.
- Run ProtoGen.bat to generate C# source files into `./ProtoGenFiles/rx.xx.x`
- Update OpGenerator project to add the newly generated C# files to the project.