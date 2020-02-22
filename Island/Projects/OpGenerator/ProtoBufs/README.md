# ProtoBufs

This directory include protocol buffer files that describe the definition of ops and apis. They are copied from `$(TF_PATH)/tensorflow/core/framework/` and put into respective sub-directory with dirname suffixed with version.

The original .proto files include relative path like `import "tensorflow/core/framework/types.proto`. The copied .proto files have those relative paths removed, directly referencing the local file in this folder as `import "types.proto`

- r1.15.0
Directory for tensorflow release version 1.15.0