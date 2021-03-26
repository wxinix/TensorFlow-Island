// MIT License
// Copyright (c) 2019-2021 Wuping Xin.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*!
  \brief This sample provides a simple wrap-up of TensorFlow C APIs.
*/
namespace TensorFlow.Island.Api.Helpers;

uses
  TensorFlow.Island.Api;

  method DeallocateBuffer(aData: ^Void; aSize: UInt64);
  begin
    free(aData);
  end;

  method ReadBufferFromFile(const aFile: not nullable String): ^TF_Buffer;
  begin
    using fs := new FileStream(aFile, FileMode.Open, FileAccess.Read) do begin
      if fs.Length < 1 then 
        exit nil;
      var data := malloc(fs.Length);
      fs.Read(data, fs.Length);
      result := TF_NewBuffer();
      result^.data := data;
      result^.length := fs.Length;
      result^.data_deallocator := @DeallocateBuffer;
    end;
  end;

  { 
    C-API functions TF_StringDecode, TF_StringEncode, and TF_StringEncodedSize are no longer
    relevant and have been removed in TensorFlow 2.4.0
  }
  method ScalarStringTensor(const aStr: ^AnsiChar; aStatus: ^TF_Status): ^TF_Tensor;
  begin
    const TF_TSTRING_SIZE: Integer = 24;
    var strlen := lstrlenA(aStr);
    result := TF_AllocateTensor(TF_DataType.TF_STRING, nil, 0, TF_TSTRING_SIZE);
    var data := ^AnsiChar(TF_TensorData(result));
    TF_TString_Init(^TF_TString(data));
    TF_TString_Copy(^TF_TString(data), aStr, strlen);
  end;  

  method LoadGraph(const aGraphPath: String; const aCheckPointPrefix: String; aStatus: ^TF_Status := nil): ^TF_Graph;
  begin
    if not File.Exists(aGraphPath) then exit nil;
    var buffer := ReadBufferFromFile(aGraphPath);
    if not assigned(buffer) then
      exit nil;

    var deleteStatus := false;
    if not assigned(aStatus) then begin
      deleteStatus := true;
      aStatus := TF_NewStatus();
    end;

    var session: ^TF_Session := nil;
    var checkpointTensor: ^TF_Tensor := nil;

    try
      result := TF_NewGraph();
      var opts := TF_NewImportGraphDefOptions();
      TF_GraphImportGraphDef(result, buffer, opts, aStatus);
      TF_DeleteImportGraphDefOptions(opts);
      TF_DeleteBuffer(buffer);

      var code := TF_GetCode(aStatus);
      if code <> TF_Code.TF_OK then begin
        TF_DeleteGraph(result);
        exit nil;
      end;

      if String.IsNullOrEmpty(aCheckPointPrefix) then
        exit;

      checkpointTensor := ScalarStringTensor(aCheckPointPrefix.ToAnsiChars(true), aStatus);
      code := TF_GetCode(aStatus);
      if code <> TF_Code.TF_OK then begin
        TF_DeleteGraph(result);
        exit nil;
      end;      

      var input: TF_Output := new TF_Output(
        oper := TF_GraphOperationByName(result, String('save/Const').ToAnsiChars(true)),
        index := 0);

      var restoreOp := TF_GraphOperationByName(result, String('save/restore_all').ToAnsiChars(true));

      session := CreateSession(result);
      if not assigned(session) then begin
        TF_DeleteGraph(result);
        exit nil;
      end;

      TF_SessionRun(
        session,
        nil,
        @input,
        @checkpointTensor,
        1,
        nil,
        nil,
        0,
        @restoreOp,
        1,
        nil,
        aStatus);

      code := TF_GetCode(aStatus);
      if code <> TF_Code.TF_OK then begin
        TF_DeleteGraph(result);
        exit nil;
      end;
    finally
      if deleteStatus then 
        TF_DeleteStatus(aStatus);
      if assigned(checkpointTensor) then 
        TF_DeleteTensor(checkpointTensor);
      if assigned(session) then 
        DeleteSession(session);
    end;
  end;

  method LoadGraph(const aGraphPath: String; aStatus: ^TF_Status = nil): ^TF_Graph;
  begin
    result := LoadGraph(aGraphPath, '', aStatus);
  end;

  method DeleteGraph(aGraph: ^TF_Graph);
  begin
    if assigned(aGraph) then 
      TF_DeleteGraph(aGraph);
  end;

  method CreateSession(aGraph: ^TF_Graph; aOptions: ^TF_SessionOptions; aStatus: ^TF_Status = nil): ^TF_Session;
  begin
    var deleteStatus := false;
    var deleteOpts := false;

    try
      if not assigned(aStatus) then begin
        deleteStatus := true;
        aStatus := TF_NewStatus();
      end;

      if not assigned(aOptions) then begin
        deleteOpts := true;
        aOptions := TF_NewSessionOptions();
      end;

      result := TF_NewSession(aGraph, aOptions, aStatus);
      if TF_GetCode(aStatus) <> TF_Code.TF_OK then begin
        DeleteSession(result);
        exit nil;
      end;
    finally
      if deleteStatus then 
        TF_DeleteStatus(aStatus);
      if deleteOpts then 
        TF_DeleteSessionOptions(aOptions);
    end;
  end;

  method CreateSession(aGraph: ^TF_Graph; aStatus: ^TF_Status := nil): ^TF_Session;
  begin
    result := CreateSession(aGraph, nil, aStatus);
  end;

  method DeleteSession(aSession: ^TF_Session; aStatus: ^TF_Status := nil): TF_Code;
  begin
    if not assigned(aSession) then 
      exit(TF_Code.TF_INVALID_ARGUMENT);

    var deleteStatus := false;
    if not assigned(aStatus) then begin
      deleteStatus := true;
      aStatus := TF_NewStatus();
    end;

    try
      TF_CloseSession(aSession, aStatus);
      var code := TF_GetCode(aStatus);
      if code <> TF_Code.TF_OK then begin
        TF_CloseSession(aSession, aStatus);
        TF_DeleteSession(aSession, aStatus);
        exit code;
      end;

      TF_DeleteSession(aSession, aStatus);
      code := TF_GetCode(aStatus);
      if code <> TF_Code.TF_OK then begin
        TF_DeleteSession(aSession, aStatus);
        exit code;
      end;
    finally
      if deleteStatus then 
        TF_DeleteStatus(aStatus);
    end;
  end;

  method RunSession(aSession: ^TF_Session; const aInputs: ^TF_Output; const aInputTensors: array of ^TF_Tensor;
    aInputSize: Int32; const aOutputs: ^TF_Output; aOutputTensors: array of ^TF_Tensor; aOutputSize: Int32;
    aStatus: ^TF_Status = nil): TF_Code;
  begin
    var exitCond := not assigned(aSession) or not assigned(aInputs) or not assigned(aInputTensors) 
      or not assigned(aOutputs) or not assigned(aOutputTensors);
    
    if exitCond then
      exit TF_Code.TF_INVALID_ARGUMENT;

    var deleteStatus := false;
    if not assigned(aStatus) then begin
      deleteStatus := true;
      aStatus := TF_NewStatus();
    end;

    try
      TF_SessionRun(
        aSession,
        nil,
        aInputs,
        aInputTensors,
        aInputSize,
        aOutputs,
        aOutputTensors,
        aOutputSize,
        nil,
        0,
        nil,
        aStatus);
      result := TF_GetCode(aStatus);
    finally
      if deleteStatus then TF_DeleteStatus(aStatus);
    end;
  end;

  method RunSession(aSession: ^TF_Session; const aInputs: List<TF_Output>; const aInputTensors: List<^TF_Tensor>;
    const aOutputs: List<TF_Output>; aOutputTensors: List<^TF_Tensor>; aStatus: ^TF_Status := nil): TF_Code;
  begin
    var outputTensors := aOutputTensors.ToArray;

    result := RunSession(
      aSession,
      aInputs.ToArray,
      aInputTensors.ToArray,
      aInputTensors.Count,
      aOutputs.ToArray,
      outputTensors,
      aOutputTensors.Count,
      aStatus);

    aOutputTensors.Clear;
    aOutputTensors.AddRange(outputTensors);
  end;

  method CreateTensor(aDataType: TF_DataType; const aDims: array of int64_t;   aDimsSize: Int32; const aData: ^Void;
    aDataByteSize: UInt64): ^TF_Tensor;
  begin
    result := CreateEmptyTensor(aDataType, aDims, aDimsSize, aDataByteSize);
    if not assigned(result) then exit;

    var tensorData := TF_TensorData(result);
    if not assigned(tensorData) then begin
      TF_DeleteTensor(result);
      result := nil;
      exit;
    end;

    aDataByteSize := Math.Min(aDataByteSize, TF_TensorByteSize(result));
    if assigned(aData) and (aDataByteSize > 0) then begin
      memcpy(tensorData, aData, aDataByteSize);
    end;
  end;

  method CreateTensor<T>(aDataType: TF_DataType; const aDims: List<int64_t>; const aData: List<T>): ^TF_Tensor;
  begin
    result := CreateTensor(
      aDataType,
      aDims.ToArray,
      aDims.Count,
      aData.ToArray,
      aData.Count * sizeOf(T));
  end;

  method CreateEmptyTensor(aDataType: TF_DataType; aDims: array of int64_t; aDimSize: Int32; aDataByteSize: UInt64 := 0): ^TF_Tensor;
  begin
    if not assigned(aDims) then 
      exit nil;
    result := TF_AllocateTensor(aDataType, aDims, aDimSize, aDataByteSize);
  end;

  method CreateEmptyTensor(aDataType: TF_DataType;const aDims: List<int64_t>; aDataByteSize: UInt64 := 0): ^TF_Tensor;
  begin
    CreateEmptyTensor(aDataType, aDims.ToArray, aDims.Count, aDataByteSize);
  end;

  method DeleteTensor(aTensor: ^TF_Tensor);
  begin
    if assigned(aTensor) then
      TF_DeleteTensor(aTensor);
  end;

  method DeleteTensors(const aTensors: List<^TF_Tensor>);
  begin
    for tensor in aTensors do
      DeleteTensor(tensor);
  end;

  method SetTensorData(aTensor: ^TF_Tensor; const aData: ^Void; aDataByteSize: UInt64): Boolean;
  begin
    var tensorData := TF_TensorData(aTensor);
    aDataByteSize := Math.Min(aDataByteSize, TF_TensorByteSize(aTensor));

    if assigned(tensorData) and assigned(aData) and (aDataByteSize > 0) then begin
      memcpy(tensorData, aData, aDataByteSize);
      exit true;
    end
    else begin
      exit false;
    end;
  end;

  method SetTensorData<T>(aTensor: ^TF_Tensor; const aData: List<T>): Boolean;
  begin
    result := SetTensorData(aTensor, aData.ToArray, aData.Count * sizeOf(T));
  end;

  method GetTensorData<T>(const aTensor: ^TF_Tensor): List<T>;
  begin
    var tensorDataTypeSize := TF_DataTypeSize(TF_TensorType(aTensor));

    if (sizeOf(T) <> tensorDataTypeSize) or not assigned(TF_TensorData(aTensor)) then begin
      exit nil;
    end;

    var size := TF_TensorByteSize(aTensor) / tensorDataTypeSize;
    var data: array of T := new T[size];
    memcpy(data, TF_TensorData(aTensor), TF_TensorByteSize(aTensor));
    result := new List<T>(data);
  end;

  method GetTensorData<T>(const aTensors: List<^TF_Tensor>): List<List<T>>;
  begin
    result := new List<List<T>>(aTensors.Count);
    for tensor in aTensors do
      result.Add(GetTensorData<T>(tensor));
  end;

  method GetTensorShape(aGraph: ^TF_Graph; const var aOutput: TF_Output): List<int64_t>;
  begin
    result := new List<int64_t>;
    var status := TF_NewStatus();

    try
      var numdims := TF_GraphGetTensorNumDims(aGraph, aOutput, status);
      if (TF_GetCode(status) <> TF_Code.TF_OK) then
        exit;
      var data: array of int64_t := new int64_t[numdims];
      TF_GraphGetTensorShape(aGraph, aOutput, data, numdims, status);
      if (TF_GetCode(status) <> TF_Code.TF_OK) then
        exit;
      result.AddRange(data);
    finally
      TF_DeleteStatus(status);
    end;
  end;

  method GetTensorShape(aGraph: ^TF_Graph; const aOutputs: List<TF_Output>): List<List<int64_t>>;
  begin
    result := new List<List<int64_t>>(aOutputs.Count);
    for output in aOutputs do
      result.Add(GetTensorShape(aGraph, output));
  end;

  method CreateSessionOptions(aGpuMemoryFraction: Double; aStatus: ^TF_Status := nil): ^TF_SessionOptions;
  begin
    var deleteStatus := false;
    if not assigned(aStatus) then begin
      aStatus := TF_NewStatus();
      deleteStatus := true;
    end;

    try
      result := TF_NewSessionOptions();
      // Equivalent in Python:
      // config = tf.ConfigProto( allow_soft_placement = True )
      // config.gpu_options.allow_growth = True
      // config.gpu_options.per_process_gpu_memory_fraction = percentage
      // Create a byte-array for the serialized ProtoConfig, set the mandatory
      // bytes (first three and last four)
      var config: array[0..14] of uint8_t := [$32, $b, $9, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $20, $1, $38, $1];
      var bytes := (^uint8_t)(@aGpuMemoryFraction);
      for i: Integer := 0 to sizeOf(aGpuMemoryFraction) - 1 do begin
        config[i + 3] := bytes[i];
      end;

      TF_SetConfig(result, config, length(config), aStatus);
      if (TF_GetCode(aStatus) <> TF_Code.TF_OK) then begin
        TF_DeleteSessionOptions(result);
        exit nil;
      end;
    finally
      if deleteStatus then TF_DeleteStatus(aStatus);
    end;
  end;

  method CreateSessionOptions(aIntraOpParallelismThreads: uint8_t; aInterOpParallelismThreads: uint8_t; aStatus: ^TF_Status := nil): ^TF_SessionOptions;
  begin
    var deleteStatus := false;
    if not assigned(aStatus) then begin
      aStatus := TF_NewStatus();
      deleteStatus := true;
    end;

    try
      result := TF_NewSessionOptions();
      var config: array[0..3] of uint8_t := [$10, aIntraOpParallelismThreads, $28, aInterOpParallelismThreads];

      TF_SetConfig(result, config, length(config), aStatus);
      if (TF_GetCode(aStatus) <> TF_Code.TF_OK) then begin
        TF_DeleteSessionOptions(result);
        exit nil;
      end;
    finally
      if deleteStatus then
        TF_DeleteStatus(aStatus);
    end;
  end;

  method DataTypeToString(aDataType: TF_DataType): String;
  begin
    result := case aDataType of
      TF_DataType.TF_FLOAT:
        'TF_FLOAT';
      TF_DataType.TF_DOUBLE:
        'TF_DOUBLE';
      TF_DataType.TF_INT32:
        'TF_INT32';
      TF_DataType.TF_UINT8:
        'TF_UINT8';
      TF_DataType.TF_INT16:
        'TF_INT16';
      TF_DataType.TF_INT8:
        'TF_INT8';
      TF_DataType.TF_STRING:
        'TF_STRING';
      TF_DataType.TF_COMPLEX64:
        'TF_COMPLEX64';
      TF_DataType.TF_INT64:
        'TF_INT64';
      TF_DataType.TF_BOOL:
        'TF_BOOL';
      TF_DataType.TF_QINT8:
        'TF_QINT8';
      TF_DataType.TF_QUINT8:
        'TF_QUINT8';
      TF_DataType.TF_QINT32:
        'TF_QINT32';
      TF_DataType.TF_BFLOAT16:
        'TF_BFLOAT16';
      TF_DataType.TF_QINT16:
        'TF_QINT16';
      TF_DataType.TF_QUINT16:
        'TF_QUINT16';
      TF_DataType.TF_UINT16:
        'TF_UINT16';
      TF_DataType.TF_COMPLEX128:
        'TF_COMPLEX128';
      TF_DataType.TF_HALF:
        'TF_HALF';
      TF_DataType.TF_RESOURCE:
        'TF_RESOURCE';
      TF_DataType.TF_VARIANT:
        'TF_VARIANT';
      TF_DataType.TF_UINT32:
        'TF_UINT32';
      TF_DataType.TF_UINT64:
        'TF_UINT64';
      else 
        'Unknown';
    end;
  end;

  method CodeToString(aCode: TF_Code): String;
  begin
    result := case aCode of
      TF_Code.TF_OK:
        'TF_OK';
      TF_Code.TF_CANCELLED:
        'TF_CANCELLED';
      TF_Code.TF_UNKNOWN:
        'TF_UNKNOWN';
      TF_Code.TF_INVALID_ARGUMENT:
        'TF_INVALID_ARGUMENT';
      TF_Code.TF_DEADLINE_EXCEEDED:
        'TF_DEADLINE_EXCEEDED';
      TF_Code.TF_NOT_FOUND:
        'TF_NOT_FOUND';
      TF_Code.TF_ALREADY_EXISTS:
        'TF_ALREADY_EXISTS';
      TF_Code.TF_PERMISSION_DENIED:
        'TF_PERMISSION_DENIED';
      TF_Code.TF_UNAUTHENTICATED:
        'TF_UNAUTHENTICATED';
      TF_Code.TF_RESOURCE_EXHAUSTED:
        'TF_RESOURCE_EXHAUSTED';
      TF_Code.TF_FAILED_PRECONDITION:
        'TF_FAILED_PRECONDITION';
      TF_Code.TF_ABORTED:
        'TF_ABORTED';
      TF_Code.TF_OUT_OF_RANGE:
        'TF_OUT_OF_RANGE';
      TF_Code.TF_UNIMPLEMENTED:
        'TF_UNIMPLEMENTED';
      TF_Code.TF_INTERNAL:
        'TF_INTERNAL';
      TF_Code.TF_UNAVAILABLE:
        'TF_UNAVAILABLE';
      TF_Code.TF_DATA_LOSS:
        'TF_DATA_LOSS';
      else
        'Unknown';
    end;
  end;

  method PrintInputs(aGraph: ^TF_Graph; aOp: ^TF_Operation);
  begin
    var numInputs := TF_OperationNumInputs(aOp);
    var input: TF_Input;
    for i: Integer := 0 to numInputs - 1 do begin
      (input.oper, input.index) := (aOp, i);
      var dataType := TF_OperationInputType(input);
      writeLn($'Input:{i} type:{DataTypeToString(dataType)}');
    end;
  end;

  method PrintOutputs(aGraph: ^TF_Graph; aOp: ^TF_Operation; aStatus: ^TF_Status);
  begin
    var numOutputs := TF_OperationNumOutputs(aOp);
    var output: TF_Output;

    for i: Integer := 0 to numOutputs - 1 do begin
      (output.oper, output.index) := (aOp, i);
      var numDims := TF_GraphGetTensorNumDims(aGraph, output, aStatus);
      if TF_GetCode(aStatus) <> TF_Code.TF_OK then begin
        writeLn('Cannot get tensor dimensionality.');
        continue;
      end;

      write($' dims:{numDims}');
      if numDims <= 0 then begin
        writeLn(' []');
        continue;
      end;

      var dataType := TF_OperationOutputType(output);
      var dims: array of int64_t := new int64_t[numDims];
      write($' Output:{i} type:{DataTypeToString(dataType)}');
      TF_GraphGetTensorShape(aGraph, output, dims, numDims, aStatus);

      if TF_GetCode(aStatus) <> TF_Code.TF_OK then begin
        writeLn('Cannot get tensor shape.');
      end
      else begin
        write(' [');
        for d: Integer := 0 to numDims - 1 do begin
          write(dims[d]);
          write(if d < numDims-1 then ', ' else ']');
        end;
      end;
    end;
  end;

  method PrintTensorInfo(aGraph: ^TF_Graph; const aLayerName: String; aStatus: ^TF_Status);
  begin
    write($'Tensor:{aLayerName}');

    var op := TF_GraphOperationByName(aGraph, aLayerName.ToAnsiChars(true));
    if not assigned(op) then begin
      writeLn($'Could not get {aLayerName}');
      exit;
    end;

    var numInputs := TF_OperationNumInputs(op);
    var numOutputs := TF_OperationNumOutputs(op);
    writeLn($' inputs:{numInputs} outputs:{numOutputs}');

    try
      PrintInputs(aGraph, op);
      PrintOutputs(aGraph, op, aStatus);
    except
      on E: Exception do
        writeLn(E.Message);
    end;
  end;

  method PrintOps(aGraph: ^TF_Graph; aStatus: ^TF_Status);
  begin
    var pos: UInt64;
    var op: ^TF_Operation := TF_GraphNextOperation(aGraph, @pos);

    while (assigned(op)) do begin
      var opName := TF_OperationName(op);
      var opType := TF_OperationOpType(op);
      var device := TF_OperationDevice(op);
      var numOutputs := TF_OperationNumOutputs(op);
      var numInputs := TF_OperationNumInputs(op);

      writeLn(
        $"{pos}:{String.FromPAnsiChars(opName)} "  +
        $"type:{String.FromPAnsiChars(opType)} "   +
        $"device:{String.FromPAnsiChars(device)} " +
        $"numberInputs:{numInputs} "               +
        $"numberOutputs:{numOutputs} ");

      PrintInputs(aGraph, op);
      PrintOutputs(aGraph, op, aStatus);
      writeLn('');
      op := TF_GraphNextOperation(aGraph, @pos);
    end;
  end;

end.