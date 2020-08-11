#import "include/tensorflow/c/c_api.h"
#import "include/tensorflow/c/tf_attrtype.h"
#import "include/tensorflow/c/tf_datatype.h"
#import "include/tensorflow/c/tf_status.h"
#import "include/tensorflow/c/tf_tensor.h"

//
// Tasks
//
// in Settings, pick the appropriate Mode (Toffee or Island), SubMode and SDK for the
// import. Also set the ImportSearchPath to contain the folder or folders that congain
// your header files
//
// In this file, add #import clauses for each header you want to import
//
// Finally, drag any binaries that need to be linked to use the imported library into
// the project and make sure their type is set to "ImportLinkLibrary".
//