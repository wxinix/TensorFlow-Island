// MIT License
// Copyright (c) 2019-2020 Wuping Xin.
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

namespace TensorFlow.Island.Classes;

uses
  RemObjects.Elements.RTL,
  RemObjects.Elements.System,
  TensorFlow.Island.Api,
  TensorFlow.Island.Aspects;

type
  NotNull<T> = public not nullable T;

  ITensorFlowDisposable = public interface(IDisposable)
    property ID: NativeUInt read;
  end;

  TensorFlowDisposable = public abstract class(ITensorFlowDisposable)
  private
    fDisposed: Boolean := false;

    finalizer;
    begin
      if not fDisposed then begin
        Dispose(false);
      end;
    end;
  protected
    method Dispose(aDisposing: Boolean); virtual;
    begin
      fDisposed := true;
    end;

    method CheckAndRaiseOnDisposed;
    begin
      if fDisposed then begin
        raise new ObjectDisposedException(self);
      end
    end;

    method get_ID: NativeUInt; virtual;
    begin
      result := NativeUInt(@self);
    end;
  public
    method Dispose;
    begin
      if not fDisposed then begin
        Dispose(true);
        BoehmGC.SuppressFinalize(self);
      end;
    end;

    property ID: NativeUInt read get_ID;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowDisposableList<T> = public abstract class(TensorFlowDisposable, IEnumerable<T>)
    where T is ITensorFlowDisposable;
  private
    fDisposed: Boolean := false;
  protected
    fList: List<T>; implements IEnumerable<T>;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        for el in fList do el.Dispose();
      end;

      fList.Clear;
      inherited Dispose(aDisposing);
    end;

    constructor withCapacity(aCapacity: Integer);
    begin
      fList := new List<T>(aCapacity);
    end;

    constructor;
    begin
      fList := new List<T>;
    end;
  public
    method &Add(aItem: T);
    begin
      fList.Add(aItem);
    end;

    method ToArray: array of T;
    begin
      result := fList.ToArray;
    end;

    property Count: Integer
      read begin
        result := fList.Count;
      end;

    property Item[i: Integer]: T
      read begin
        result := fList[i];
      end; default;
  end;

  ITensorFlowObject = public interface(ITensorFlowDisposable)
    property Handle: ^Void read;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowObject<T> = public abstract class(TensorFlowDisposable, ITensorFlowObject)
  public
    type DisposeAction<T> = block(aHandle: ^T);
  private
    fOnDispose: DisposeAction<T>;
    fHandle: ^T := nil;
    fDisposed: Boolean := false;
  protected
    constructor withHandle(aHandle: NotNull<^T>) OnDispose(aAction: DisposeAction<T>);
    begin
      fHandle := aHandle;
      fOnDispose := aAction;
    end;

    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        // Derived class should call its managed object's Dispose().
      end;

      if assigned(fOnDispose) then begin
        fOnDispose(fHandle);
        fHandle := nil;
      end;

      inherited Dispose(aDisposing);
    end;

    method get_ID: NativeUInt; override;
    begin
      result := NativeInt(fHandle);
    end;
  public
    property ID: NativeUInt read get_ID;

    property Handle: ^Void
      read begin
        exit fHandle;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowObjectList<T> = public abstract class(TensorFlowDisposableList<T>)
    where T is ITensorFlowObject;
  public
    property Handles: array of ^Void
      read begin
        if fList.Count = 0 then exit nil;
        result := new ^Void[fList.Count];
        for I: Integer := 0 to fList.Count - 1 do begin
          result[I] := fList[I].Handle;
        end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Buffer = public sealed class(TensorFlowObject<TF_Buffer>)
  private
    fData: ^Void := nil;
    fNumBytes: UInt64 := 0;
    fOnDispose: DisposeAction<TF_Buffer> := aHandle->TF_DeleteBuffer(aHandle);
  public
    class method DeallocateBuffer(aData: ^Void; aSize: UInt64);
    begin
      free(aData);
    end;

    constructor withFile(aFile: NotNull<String>);
    begin
      var buf_bytes := Helper.ReadBytesFromFile(aFile);

      if assigned(buf_bytes) then begin
        fNumBytes := buf_bytes.Length;
        fData := malloc(fNumBytes);
        memcpy(fData, buf_bytes, fNumBytes);
      end else begin
        fNumBytes := 0;
        fData := nil;
      end;

      var buf_handle := TF_NewBuffer();
      buf_handle^.data := fData;
      buf_handle^.length := fNumBytes;
      buf_handle^.data_deallocator := @DeallocateBuffer;

      inherited constructor withHandle(buf_handle) OnDispose(fOnDispose);
    end;

    constructor withString(const aProtoBuf: NotNull<String>);
    begin
      var proto_len := lstrlenA(aProtoBuf.ToAnsiChars(true));
      var buf_handle: ^TF_Buffer := TF_NewBufferFromString(aProtoBuf.ToAnsiChars, proto_len);

      fData := buf_handle^.data;
      fNumBytes := buf_handle^.length;
      inherited constructor withHandle(buf_handle) OnDispose(fOnDispose);
    end;

    constructor withHandle(aHandle: ^TF_Buffer); assembly;
    begin
      fData := aHandle^.data;
      fNumBytes := aHandle^.length;
      inherited constructor withHandle(aHandle) OnDispose(fOnDispose);
    end;

    method ToArray: array of Byte;
    begin
      if fNumBytes > 0 then begin
        result := new Byte[fNumBytes];
        memcpy(result, fData, fNumBytes);
      end else begin
        result := nil;
      end;
    end;

    property NumBytes: UInt64
      read begin
        result:= fNumBytes;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Operation = public sealed class(TensorFlowObject<TF_Operation>)
  private
    fName: String;
    fGraph: Graph;
  public
    constructor withHandle(aHandle: ^TF_Operation) Name(aName: NotNull<String>) 
      Graph(aGraph: NotNull<Graph>); assembly;
    begin
      fGraph := aGraph;
      fName := aName;
      inherited constructor withHandle(aHandle) OnDispose(nil);
    end;

    method ToString: String; override;
    begin
      result := $'Operation: {Convert.UInt64ToHexString(ID, 16)}';
    end;

    property &Graph: Graph
      read begin
        result := fGraph;
      end;
    property Name: String
      read begin
        result := fName;
      end;
  end;

  OperationList = public sealed class(TensorFlowObjectList<Operation>)
  public
    constructor withCapacity(aCapacity: Integer);
    begin
      inherited constructor withCapacity(aCapacity);
    end;

    constructor;
    begin
      inherited constructor;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  OperationDescription = public sealed class(TensorFlowObject<TF_OperationDescription>)
  private
    fGraph: Graph;
    fOpType: String;
    fOperName: String;
  public
    constructor withGraph(aGraph: NotNull<Graph>) OpType(aType: NotNull<String>)
      OpName(aName: NotNull<String>);
    begin
      fOpType := aType;
      fOperName := aName;
      fGraph := aGraph;

      var op_desc_handle := TF_NewOperation(aGraph.Handle, aType.ToAnsiChars(true),
        aName.ToAnsiChars(true));
      // OnDispose nil, TF_FinishOption will delete OperationDescription.
      inherited constructor withHandle(op_desc_handle) OnDispose(nil);
    end;

    method SetDevice(aDevice: not nullable String);
    begin
      TF_SetDevice(Handle, aDevice.ToAnsiChars(true));
    end;

    method AddInput(aInput: NotNull<Output>);
    begin
      TF_AddInput(Handle, aInput.AsTFOutput);
    end;

    method AddInputs(aInputList: NotNull<array of Output>);
    begin
      var tfOutput := new TF_Output[aInputList.Length];
      for I: Integer := 0 to aInputList.Length - 1 do begin
        tfOutput[I] := aInputList[I].AsTFOutput;
      end;

      TF_AddInputList(Handle, tfOutput, tfOutput.Length);
    end;

    method FinishOperation(aStatus: Status := nil): Tuple of (Boolean, Operation);
    begin
      using lStatus := new Status do begin
        // Desc ptr gets deleted inside TF_FinishOperation.
        var op_handle := TF_FinishOperation(Handle, lStatus.Handle);

        if lStatus.OK then begin
          result := (true, new Operation withHandle(op_handle) Name(fOperName) Graph(fGraph))
        end else begin
          result := (false, nil);
        end;

        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method SetAttr(const aName: NotNull<String>; aValue: NotNull<Object>);
    begin
      var lHashCode:= aValue.GetType.GetHashCode;
      case lHashCode of
        typeOf(Boolean).GetHashCode:
          SetAttrBool(aName, aValue as Boolean);
        typeOf(array of Boolean).GetHashCode:
          SetAttrBoolList(aName, aValue as array of Boolean);
        typeOf(Single).GetHashCode:
          SetAttrFloat(aName, aValue as Single);
        typeOf(array of Single).GetHashCode:
          SetAttrFloatList(aName, aValue as array of Single);
        typeOf(Int64).GetHashCode:
          SetAttrInt(aName, aValue as Int64);
        typeOf(array of Int64).GetHashCode:
          SetAttrIntList(aName, aValue as array of Int64);
        typeOf(Tensor).GetHashCode:
          SetAttrTensor(aName, aValue as Tensor);
        typeOf(Shape).GetHashCode:
          SetAttrShape(aName, aValue as Shape);
        typeOf(array of Shape).GetHashCode:
          SetAttrShapeList(aName,  aValue as array of Shape);
        typeOf(TensorFlowDataType).GetHashCode:
          SetAttrType(aName, TF_DataType(ord(aValue as TensorFlowDataType)));
        typeOf(String).GetHashCode:
          SetAttrStr(aName, aValue as String);
        typeOf(array of String).GetHashCode:
          SetAttrStringList(aName, aValue as array of String);
      end;
    end;

    method SetAttrBool(const aName: NotNull<String>; aValue: Boolean);
    begin
      var value: Byte := if aValue then 1 else 0;
      TF_SetAttrBool(Handle, aName.ToAnsiChars(true), value);
    end;

    method SetAttrBoolList(const aName: NotNull<String>; aList: NotNull<array of Boolean>);
    begin
      var values := new Byte[aList.Length];
      for I: Integer := 0 to aList.Length - 1 do begin
        values[I] := if aList[I] then 1 else 0;
      end;

      TF_SetAttrBoolList(Handle, aName.ToAnsiChars(true), values, values.Length);
    end;

    method SetAttrFloat(const aName: not nullable String; aValue: Single);
    begin
      TF_SetAttrFloat(Handle, aName.ToAnsiChars(true), aValue);
    end;

    method SetAttrFloatList(const aName: NotNull<String>; aList: NotNull<array of Single>);
    begin
      TF_SetAttrFloatList(Handle, aName.ToAnsiChars(true), aList, aList.Length);
    end;

    method SetAttrInt(const aName: NotNull<String>; aValue: Int64);
    begin
      TF_SetAttrInt(Handle, aName.ToAnsiChars(true), aValue);
    end;

    method SetAttrIntList(const aName: NotNull<String>; aList: NotNull<array of Int64>);
    begin
      TF_SetAttrIntList(Handle, aName.ToAnsiChars(true), aList, aList.Length);
    end;

    method SetAttrStr(const aName: NotNull<String>; aValue: NotNull<String>);
    begin
      var length := lstrlenA(aValue.ToAnsiChars(true));
      TF_SetAttrString(Handle, aName.ToAnsiChars(true), aValue.ToAnsiChars, length);
    end;

    method SetAttrStringList(const aName: NotNull<String>; aList: NotNull<array of String>);
    begin
      var num_values := aList.Length;
      var values: array of array of AnsiChar := new array of AnsiChar[num_values];
      var lengths: array of UInt64 := new UInt64[num_values];

      for I: Integer := 0 to num_values - 1 do begin
        // No null terminator, because length is explicitly given below.
        values[I] := aList[I].ToAnsiChars;
        lengths[I] := aList[I].Length;
      end;

      TF_SetAttrStringList(Handle, aName.ToAnsiChars(true), ^^Void(values),
        lengths, num_values);
    end;

    method SetAttrType(const aName: NotNull<String>; aType: TF_DataType);
    begin
      TF_SetAttrType(Handle, aName.ToAnsiChars(true), aType);
    end;

    method SetAttrTypeList(const aName: NotNull<String>; aTypeList: NotNull<array of TF_DataType>);
    begin
      TF_SetAttrTypeList(Handle, aName.ToAnsiChars(true), aTypeList, aTypeList.Length);
    end;

    method SetAttrTensor(const aName: NotNull<String>; aTensor: NotNull<Tensor>;
      aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrTensor(Handle, aName.ToAnsiChars(true), aTensor.Handle, lStatus.Handle);
        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method SetAttrShape(const aName: NotNull<String>; aShape: NotNull<Shape>);
    begin
      TF_SetAttrShape(Handle, aName.ToAnsiChars(true), aShape.ToArray, aShape.NumDims);
    end;

    method SetAttrShapeList(const aName: NotNull<String>; aList: NotNull<array of Shape>);
    begin
      var num_shapes: Int32 := aList.Length;
      var dims: array of array of Int64 := new array of Int64[num_shapes];
      var num_dims: array of Int32 := new Int32[num_shapes];

      for I: Integer := 0 to num_shapes - 1 do begin
        dims[I] := aList[I].ToArray;
        num_dims[I] := aList[I].NumDims;
      end;

      TF_SetAttrShapeList(Handle, aName.ToAnsiChars(true), ^^Int64(dims), num_dims, num_shapes);
    end;

    property OpType: String
      read begin
        result := fOpType;
      end;

    property OperName: String
      read begin
        result := fOperName;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Status = public sealed class(TensorFlowObject<TF_Status>)    
  public
    [ThreadLocal] 
    class var &Default: Status := new Status;
  public
    constructor;
    begin
      var status_handle := TF_NewStatus();
      inherited constructor withHandle(status_handle) OnDispose(aHandle->TF_DeleteStatus(aHandle));
    end;

    method SetCode(aCode: TensorFlowCode) withMessage(const aMsg: String);
    begin
      if not String.IsNullOrEmpty(aMsg) then
        TF_SetStatus(Handle, TF_Code(ord(aCode)), aMsg.ToAnsiChars(true))
      else
        TF_SetStatus(Handle, TF_Code(ord(aCode)), nil);
    end;

    method ToString: String; override;
    begin
      exit $'[Status: Code={Code.ToString}, Message={Message}]';
    end;

    class method ForwardOrCreate(aIncoming: Status): Status;
    begin
      result := if assigned(aIncoming) then aIncoming else new Status;
    end;

    property OK: Boolean
      read begin
        result := ord(Code) = ord(TF_Code.TF_OK);
      end;

    property Code: TensorFlowCode
      read begin
        result := TensorFlowCode(ord(TF_GetCode(Handle)));
      end;

    property Message: String
      read begin
        result := String.FromPAnsiChars(TF_Message(Handle));
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Scope = public sealed class(TensorFlowDisposable)
  public
    type ScopeRestoreAction = block(const aScopeToRestore: NotNull<String>);
  private
    fRestoreAction: ScopeRestoreAction;
    fSavedScope: String;
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
      end;

      if assigned(fRestoreAction) then begin
        fRestoreAction(fSavedScope);
        fRestoreAction := nil;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withScopeToSave(const aScope: NotNull<String>)
      RestoreAction(aAction: ScopeRestoreAction);
    begin
      fSavedScope := aScope;
      fRestoreAction := aAction;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Shape = public sealed class(TensorFlowDisposable)
  private
    fDims: array of Int64;
    fNumDims: Int32;
    fSize: UInt64;
  public
    constructor withDims(aDims: array of Int64); // aDims can be nil.
    begin
      fNumDims := if assigned(aDims) then aDims.Length else 0;
      fDims := aDims;

      if NumDims> 0 then begin
        fSize := 1;
        for I: Integer := 0 to fNumDims - 1 do fSize := fSize * fDims[I];
      end else begin
        fSize := 1;
      end;
    end;

    method ToArray: array of Int64;
    begin
      result := fDims;
    end;

    method ToString: String; override;
    begin
      var dims_str: String := '';
      for I: Integer := 0 to fNumDims - 1 do begin
        dims_str :=
          if (I < fNumDims - 1) then
            dims_str + $'Dimension({fDims[I]}), '
          else
            dims_str + $'Dimension({fDims[I]})';
      end;
      result := $'TensorShape([{dims_str}])';
    end;

    operator Implicit(aDims: array of Int64): Shape;
    begin
      result := new Shape withDims(aDims);
    end;

    property Dim[aIndex: Int32]: Int64
      read begin
        if (fNumDims > 0) and (0 <= aIndex < fNumDims) then begin
          result := fDims[aIndex]
        end else begin
          raise new InvalidShapeDimIndexException(aIndex, fNumDims);
        end;
      end;

    property NumDims: Int32
      read begin
        result := fNumDims;
      end;

    property Size: UInt64
      read begin
        result := fSize;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Output = public sealed class(TensorFlowDisposable)
  private
    fIndex: Integer;
    fOper: Operation;
  public
    constructor withOp(aOp: NotNull<Operation>) Index(aIndex: Integer = 0);
    begin
      fIndex := aIndex;
      fOper := aOp;
    end;

    method AsTFOutput: TF_Output; assembly;
    begin
      result.oper  := self.Oper.Handle;
      result.index := self.Index;
    end;

    method ToString: String; override;
    begin
      result := $'[Output: Operation = {self.Oper.ToString} Index = {self.Index}]';
    end;

    property &Index: Integer
      read begin
        result := fIndex;
      end;

    property NumConsumers: Integer
      read begin
        result := TF_OperationOutputNumConsumers(AsTFOutput());
      end;

    property Oper: Operation
      read begin
        result := fOper;
      end;

    property DataType: TensorFlowDataType
      read begin
        result := TensorFlowDataType(ord(TF_OperationOutputType(AsTFOutput)));
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  OutputList = public sealed class(TensorFlowDisposableList<Output>)
  assembly
    method AsTFOutputs: array of TF_Output;
    begin
      result := Helper.ToArray(self.ToArray);
    end;
  end;

  InputList = public sealed class(TensorFlowDisposableList<Output>)
  assembly
    method AsTFInputs: array of TF_Output;
    begin
      result := Helper.ToArray(self.ToArray);
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Graph = public sealed partial class(TensorFlowObject<TF_Graph>)
  private
    fCurrentScope: NotNull<String> := '';
    fNamesCache: Dictionary<String, Integer> := new Dictionary<String, Integer>;
    fPendingInitVars: OperationList := new OperationList;
    fDisposed: Boolean := false;

    method MakeUniqueName(const aName: NotNull<String>): String;
    begin
      var seqid := 0;

      if fNamesCache.ContainsKey(aName) then begin
        seqid := fNamesCache[aName];
        inc(seqid);
        fNamesCache[aName] := seqid;
      end else begin
        fNamesCache.Add(aName, seqid);
      end;

      result := $'{aName}_{seqid}';
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fPendingInitVars.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor;
    begin
      var graph_handle := TF_NewGraph();
      inherited constructor withHandle(graph_handle) OnDispose(aHandle->TF_DeleteGraph(aHandle));
    end;

    method WithScope(aNewScope: NotNull<String>): Scope;
    begin
      result := new Scope withScopeToSave(fCurrentScope)
        RestoreAction(aScopeToRestore->begin fCurrentScope := aScopeToRestore end);

      if String.IsNullOrEmpty(CurrentScope) then begin
        fCurrentScope := aNewScope
      end else begin
        fCurrentScope := fCurrentScope + '/' + aNewScope;
      end
    end;

    method GetOperationByName(const aOpName: NotNull<String>): Tuple of (Boolean, Operation);
    begin
      var op_handle := TF_GraphOperationByName(Handle, aOpName.ToAnsiChars(true));

      if assigned(op_handle) then begin
        result := (true,
          new Operation withHandle(op_handle) Name(aOpName) Graph(self));
      end else begin
        result := (false, nil);
      end;
    end;

    method GetShape(aOutput: NotNull<Output>; aStatus: Status := nil): Tuple of (Boolean, Shape);
    begin
      using lStatus := new Status do begin
        var nativeOut := aOutput.AsTFOutput;
        var numDims := TF_GraphGetTensorNumDims(Handle, nativeOut, lStatus.Handle);

        if (not lStatus.OK) then begin
          result := (false, nil);
        end else begin
          if numDims > 0 then begin
            var dims := new Int64[numDims];
            TF_GraphGetTensorShape(Handle, nativeOut, dims, numDims, lStatus.Handle);
            if lStatus.OK then begin
              result := (true, new Shape withDims(dims));
            end else begin
              result := (false, nil);
            end;
          end else begin
            result := (true, new Shape withDims(nil));
          end;
        end;

        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method MakeName(const aOpType, aOpName: NotNull<String>): String;
    begin
      var lOpName :=
        if String.IsNullOrEmpty(aOpName) then
          aOpType
        else
          aOpName;

      var name :=
        if String.IsNullOrEmpty(CurrentScope) then
          $'{lOpName}'
        else
          $'{CurrentScope}/{lOpName}';

      result := MakeUniqueName(name);
    end;

    method AddInitVariable(aOp: NotNull<Operation>);
    begin
      for each el in fPendingInitVars do begin
        if el.Equals(aOp) then exit;
      end;

      fPendingInitVars.Add(aOp);
    end;

    property CurrentScope: String
      read begin
        result := fCurrentScope;
      end;

    property GlobalVariableInitializer: array of ^TF_Operation
      read begin
        result := fPendingInitVars.Handles;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorData = public class(TensorFlowDisposable)
  private
    fDisposed: Boolean := false;
  protected
    fBytes: ^Void;
    fDataType: TensorFlowDataType;
    fManaged: Boolean;
    fNumBytes: UInt64;
    fShape: Shape;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fShape.Dispose;
      end;

      if fManaged then begin
        free(fBytes);
      end;

      inherited Dispose(aDisposing);
    end;

    constructor; empty;
  public
    class method DeallocateTensorData(aData: ^Void; aLen: UInt64; aArgs: ^Void);
    begin
      // This does nothing since we use Dispose pattern to manage the data.
      // Also be aware - this user defined deallocator may be IMMEDIATELY called
      // inside TF_NewTensor, as see fit by TensorFlow. This means, later by
      // calling TF_DeleteTensor, this user defined deallocator will not be called.
      // In any case, this class will manage the user-allocated memory if it is
      // constructed using constructor withData. If this class is constructed
      // using constructor withTensorHandle, then the associated data will be managed
      // by whoever creates that raw Tensor pointer.
    end;

    constructor withTensorHandle(aTensorHandle: NotNull<^TF_Tensor>); assembly;
    begin
      fBytes := TF_TensorData(aTensorHandle);
      fDataType := TensorFlowDataType(ord(TF_TensorType(aTensorHandle)));
      fNumBytes := TF_TensorByteSize(aTensorHandle);

      var lNumDims := TF_NumDims(aTensorHandle);
      var lDims: array of Int64;
      if lNumDims > 0 then begin
        lDims := new Int64[lNumDims];
        for I: Integer := 0 to lNumDims - 1 do begin
          lDims[I] := TF_Dim(aTensorHandle, I);
        end;
      end else begin
        lDims := nil;
      end;

      fManaged := false;
      fShape := new Shape withDims(lDims);
    end;

    constructor (aBytes: ^Void; aDataType: TensorFlowDataType; aNumBytes: Int64;
      aShp: NotNull<Shape>; aManaged: Boolean); private;
    begin
      fBytes := aBytes;
      fDataType := aDataType;
      fNumBytes := aNumBytes;
      fShape := aShp;
      fManaged := aManaged;
    end;

    /// <summary>
    /// Move the internal data to a new object, while flagging the old object
    /// as disposed. This emulates C++/14 move constructor.
    /// </summary>
    /// <returns></returns>
    method Move: TensorData;
    begin
      fDisposed := true;
      result := new TensorData(fBytes, fDataType, fNumBytes, fShape, fManaged);
    end;

    property Bytes: ^Void
      read begin
        result := fBytes;
      end;

    property NumBytes: UInt64
      read begin
        result := fNumBytes
      end;

    property DataType: TensorFlowDataType
      read begin
        result := fDataType
      end;

    property &Shape: Shape
      read begin
        result := fShape
      end;
  end;

  TensorData<T> = public sealed class(TensorData)
  private
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        //
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withValues(aVals: NotNull<array of T>) Dims(aDims: array of Int64);
    begin
      fDataType := Helper.ToTensorFlowDataType(typeOf(T));
      fShape := new Shape withDims(aDims);
      fManaged := true;

      if aVals.Length <> fShape.Size then begin
        raise new InvalidTensorDataSizeException withDataSize(aVals.Length) DimSize(fShape.Size);
      end;

      if fDataType <> TensorFlowDataType.String then begin
        fNumBytes := TF_DataTypeSize(TF_DataType(ord(fDataType))) * aVals.Length;
        fBytes := malloc(fNumBytes);
        if aVals.Length = 1 then begin
          (^T(fBytes))^ := aVals[0];
        end else begin
          memcpy(fBytes, aVals, fNumBytes);
        end;
      end else begin // Special handling for TF_STRING
        fNumBytes := 0;
        for I: Integer := 0 to aVals.Length - 1 do begin
          var str := Helper.EncodeString(String(aVals[I]));
          aVals[I] := T(str); // Replace with encoded string.
          fNumBytes := fNumBytes + str.Length + sizeOf(UInt64); // Offset is UInt64
        end;

        fBytes := malloc(fNumBytes);
        var num_offsets := aVals.Length; // num_offsets equal to aVals.Length
        var offsets := new UInt64[num_offsets];
        var offsets_region_size := num_offsets * sizeOf(UInt64);
        var nbytes := 0;

        for I: Integer := 0 to aVals.Length - 1 do begin
          offsets[I] := offsets_region_size + nbytes;
          memcpy(^Void(^Byte(fBytes) + offsets[I]), String(aVals[I]).ToAnsiChars, String(aVals[I]).Length);
          nbytes := nbytes + String(aVals[I]).Length;
        end;

        if fShape.NumDims = 0 then begin
          memset(fBytes, 0, sizeOf(UInt64));
        end else begin
          memcpy(fBytes, offsets, offsets_region_size);
        end;
      end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Tensor = public sealed class(TensorFlowObject<TF_Tensor>)
  private
    fData: TensorData;
    fDisposed: Boolean := false;

    class method ConvertToTensor<T>(aVals: NotNull<array of T>): Tensor; overload;
    begin
      var data := new TensorData<T> withValues(aVals) Dims([aVals.Length]);
      result := new Tensor withData(data.Move); // Moved data for Tensor's ownership.
    end;

    class method ConvertToTensor<T>(aVals: NotNull<array of NotNull<array of T>>): Tensor; overload;
    begin
      var height := aVals.Length;
      var width  := aVals[0].Length;

      for I: Integer := 0 to height - 1 do begin
        if aVals[I].Length <> width then begin
          raise new InvalidRectangularTensorData(
            $'Array [Rectangular array height={height} width={width}].' +
            $'Invalid row {I} width={aVals[I].Length}.'
          );
        end;
      end;

      var arr: array of T := new T[height * width];
      for I: Integer := 0 to height - 1 do begin
        memcpy(@arr[I * width], @aVals[I][0], width * sizeOf(T));
      end;

      var data := new TensorData<T> withValues(arr) Dims([height, width]);
      result := new Tensor withData(data.Move); // Moved data for Tensor's ownership.
    end;

    class method ConvertToTensor<T>(aValue: T): Tensor; overload;
    begin
      var data := new TensorData<T> withValues([aValue]) Dims(nil);
      result := new Tensor withData(data.Move); // Moved data for Tensor's ownership.
    end;

  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fData.Dispose; // The internal data gets released here.
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withData(aData: NotNull<TensorData>);
    begin
      var tensor_handle := TF_NewTensor(
        TF_DataType(ord(aData.DataType)),
        aData.Shape.ToArray,
        aData.Shape.NumDims,
        aData.Bytes, // Must be raw bytes; cannot be managed array.
        aData.NumBytes,
        @TensorData.DeallocateTensorData, // does nothing.
        nil);

      if not assigned(tensor_handle) then begin
        raise new TensorCreateException(aData.DataType);
      end;

      fData := aData;
      inherited constructor withHandle(tensor_handle) OnDispose(aHandle->TF_DeleteTensor(aHandle));
    end;

    constructor withHandle(aHandle: not nullable ^TF_Tensor); assembly;
    begin
      var lData: TensorData := new TensorData withTensorHandle(aHandle);
      constructor withData(lData);
    end;

    operator Implicit(aValue: Boolean): Tensor;
    begin
      result := ConvertToTensor<Boolean>(aValue);
    end;

    operator Implicit(aValue: Byte): Tensor;
    begin
      result := ConvertToTensor<Byte>(aValue);
    end;

    operator Implicit(aValue: Int16): Tensor;
    begin
      result := ConvertToTensor<Int16>(aValue);
    end;

    operator Implicit(aValue: Integer): Tensor;
    begin
      result := ConvertToTensor<Integer>(aValue);
    end;

    operator Implicit(aValue: Int64): Tensor;
    begin
      result := ConvertToTensor<Int64>(aValue);
    end;

    operator Implicit(aValue: Single): Tensor;
    begin
      result := ConvertToTensor<Single>(aValue);
    end;

    operator Implicit(aValue: Double): Tensor;
    begin
      result := ConvertToTensor<Double>(aValue);
    end;

    operator Implicit(aValue: NotNull<String>): Tensor;
    begin
      result := ConvertToTensor<String>(aValue);
    end;

    operator Implicit(aValues: NotNull<array of Boolean>): Tensor;
    begin
      result := ConvertToTensor<Boolean>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Byte>): Tensor;
    begin
      result := ConvertToTensor<Byte>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Int16>): Tensor;
    begin
      result := ConvertToTensor<Int16>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Integer>): Tensor;
    begin
      result := ConvertToTensor<Integer>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Int64>): Tensor;
    begin
      result := ConvertToTensor<Int64>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Single>): Tensor;
    begin
      result := ConvertToTensor<Single>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Double>): Tensor;
    begin
      result := ConvertToTensor<Double>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of String>): Tensor;
    begin
      result := ConvertToTensor<String>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Byte>>): Tensor;
    begin
      result := ConvertToTensor<Byte>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Boolean>>): Tensor;
    begin
      result := ConvertToTensor<Boolean>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Int16>>): Tensor;
    begin
      result := ConvertToTensor<Int16>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Integer>>): Tensor;
    begin
      result := ConvertToTensor<Integer>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Int64>>): Tensor;
    begin
      result := ConvertToTensor<Int64>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Single>>): Tensor;
    begin
      result := ConvertToTensor<Single>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Double>>): Tensor;
    begin
      result := ConvertToTensor<Double>(aValues);
    end;

    /// <summary>
    /// Convert tensor data to typed scalar value. If the underlying TensorFlow data
    /// type does not match the type parameter, no cast will be performed and nil returned.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    method AsScalar<T>: Tuple of (Boolean, nullable T);
    begin
      var valid_type := (fData.DataType = Helper.ToTensorFlowDataType(typeOf(T)) RaiseOnInvalid(false));

      if not (IsScalar and valid_type) then begin
        result := (false, nil);
      end else begin
        if (fData.DataType = TensorFlowDataType.String) then begin
          var str: String := Helper.DecodeString(String.FromPAnsiChars(
            ^AnsiChar(^Byte(fData.Bytes) + sizeOf(UInt64)), fData.NumBytes - sizeOf(UInt64)));
          result := (true, T(str));
        end else begin
          var value: T := (^T(fData.Bytes))^;
          result := (true, value);
        end;
      end;
    end;

    /// <summary>
    /// Convert tensor data to typed array. If the underlying TensorFlow data type
    /// does not match the type parameter, no cast will be performed and nil array returned.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    method AsArray<T>: Tuple of (Boolean, array of T);
    begin
      var valid_type := (fData.DataType = Helper.ToTensorFlowDataType(typeOf(T)) RaiseOnInvalid(false));

      if not (not IsScalar and valid_type) then begin
        result := (false, nil);
      end else begin
        if (fData.DataType = TensorFlowDataType.String) then begin
          var offsets := new UInt64[fData.Shape.Size];
          memcpy(offsets, fData.Bytes, sizeOf(UInt64) * fData.Shape.Size);

          var str_list: List<String> := new List<String>;
          for I: Integer := 0 to offsets.Length - 1 do begin
            var nbytes := if (I <> offsets.Length - 1) then (offsets[I + 1] - offsets[I]) else (fData.NumBytes - offsets[I]);
            var str := String.FromPAnsiChars(^AnsiChar(^Byte(fData.Bytes) + offsets[I]), nbytes);
            str := Helper.DecodeString(str);
            str_list.Add(str);
          end;

          // From String List to array of String. This is to hush compiler.
          var str_arr: array of T := new T[str_list.Count];
          for I: Integer := 0 to str_list.Count - 1 do begin
            str_arr[I] := T(str_list[I]);
          end;
          result := (true, str_arr);
        end else begin
          var values: array of T := new T[fData.Shape.Size];
          memcpy(values, fData.Bytes, fData.NumBytes);
          result := (true, values);
        end;
      end;
    end;

    /// <summary>
    /// Convert the tensor data to string array. If the underlying TensorFlow
    /// data type is TF_STRING, no cast will be performed. Otherwise, numerical
    /// type will be converted to String according to the specified aDecimalDigits
    /// parameter.
    /// </summary>
    /// <param name="aDecimalDigits"></param>
    /// <returns></returns>
    method ConvertDataToStrings(aDecimalDigits: Integer := 1): Tuple of (Boolean, array of String); private;
    begin
      // Local method to convert typed data array to string array.
      method _DoConvertDataToStrings<T>(_aData: array of T): array of String;
      begin
        if not assigned(_aData) then begin
          result := nil
        end else begin
          result := new String[_aData.Length];
        end;

        for I: Integer := 0 to _aData.Length - 1 do begin
          if (typeOf(T) = typeOf(Single)) or (typeOf(T) = typeOf(Double)) then
            result[I] := Double(_aData[I]).ToString(aDecimalDigits)
          else
            result[I] := _aData[I].ToString;
        end;
      end;

      var str_arr := case fData.DataType of
        TensorFlowDataType.Bool   : _DoConvertDataToStrings<Boolean>(AsArray<Boolean>()[1]);
        TensorFlowDataType.UInt8  : _DoConvertDataToStrings<Byte>   (AsArray<Byte>   ()[1]);
        TensorFlowDataType.UInt16 : _DoConvertDataToStrings<UInt16> (AsArray<UInt16> ()[1]);
        TensorFlowDataType.UInt32 : _DoConvertDataToStrings<UInt32> (AsArray<UInt32> ()[1]);
        TensorFlowDataType.UInt64 : _DoConvertDataToStrings<UInt64> (AsArray<UInt64> ()[1]);
        TensorFlowDataType.Int8   : _DoConvertDataToStrings<Int8>   (AsArray<Int8>   ()[1]);
        TensorFlowDataType.Int16  : _DoConvertDataToStrings<Int16>  (AsArray<Int16>  ()[1]);
        TensorFlowDataType.Int32  : _DoConvertDataToStrings<Int32>  (AsArray<Int32>  ()[1]);
        TensorFlowDataType.Int64  : _DoConvertDataToStrings<Int64>  (AsArray<Int64>  ()[1]);
        TensorFlowDataType.Float  : _DoConvertDataToStrings<Single> (AsArray<Single> ()[1]);
        TensorFlowDataType.Double : _DoConvertDataToStrings<Double> (AsArray<Double> ()[1]);
        TensorFlowDataType.String : AsArray<String>()[1];
      else nil; end;

      result := (assigned(str_arr), str_arr);
    end;

    method Print(const aDecimalDigits: Integer = 1; const aMaxWidth: Integer = 8): String;
    begin
      const cMaxBytes = 1000; // Tensor cannot exceed this limit in order to print.
      const cAllowedTypes = TensorFlowNumericalTypes + [TensorFlowDataType.String, TensorFlowDataType.Bool];
      // Validate max bytes and allowed types.
      if fData.NumBytes > cMaxBytes then
        exit 'Tensor has {fData.NumBytes} bytes. Too large (>{cMaxBytes}) to print.';
      if not (fData.DataType in cAllowedTypes) then
        exit $'Tensor (dtype={fData.DataType.ToString}) cannot print.';

      // Convert the tensor data to str_arr. Numerical type will be cast based on aDecimalDigits.
      var (success, str_arr) := ConvertDataToStrings(aDecimalDigits);
      if not success then exit 'Cannot print tensor.';

      // Put high_dims item into one line [v_1, v_2, ..,v_high_dim], inserting each into a seperate str_list
      var high_dim := fData.Shape.Dim[fData.Shape.NumDims - 1];
      var str_list := new List<String>(fData.Shape.Size/high_dim);
      var str: String := '';
      for I: Integer := 0 to str_arr.Length - 1 do begin
        if str_arr[I].Length >= aMaxWidth then exit $'Data item {str_arr[I]} exceeds max width {aMaxWidth}';
        str := str + str_arr[I].PadStart(aMaxWidth, ' ');
        if ((I + 1) mod high_dim = 0) then begin // At each "high-dim" check point
          str_list.Add($'[{str}]'); // Prefix [ and suffix ], then insert into a seperate str_list
          str := '';
        end;
      end;

      // Processt the new str_list: prefix [ and suffix ] with proper white-space.
      for strIndex: Integer := 0 to str_list.Count - 1 do begin
        for dimIndex: Integer := fData.Shape.NumDims - 2 downto 0 do begin
          if ((strIndex + 1) mod fData.Shape.Dim[dimIndex]) =  0 then
            str_list[strIndex] := $'{str_list[strIndex]} ]'
          else
            str_list[strIndex] := $'{str_list[strIndex]}  ';

          if ((strIndex + 1) mod fData.Shape.Dim[dimIndex]) =  1 then
            str_list[strIndex] := $'[ {str_list[strIndex ]}'
          else
            str_list[strIndex] := $'  {str_list[strIndex]}';
        end;
      end;

      // Concat all strings into one string, using line feed.
      result := str_list.JoinedString(#10);
    end;

    property Data: TensorData
      read begin
        result := fData;
      end;

    property IsScalar: Boolean
      read begin
        result := fData.Shape.NumDims = 0;
      end;
  end;

  TensorList = public class(TensorFlowObjectList<Tensor>)
  public
    constructor withCapacity(aCapacity: Integer);
    begin
      inherited constructor withCapacity(aCapacity);
    end;

    constructor;
    begin
      inherited constructor;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Session = public sealed class(TensorFlowObject<TF_Session>)
  private
    fGraph: Graph:= nil; // Created in constructor.
    fRunner: SessionRunner := nil; // Delayed creation upon access.
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fGraph.Dispose;
        fRunner:Dispose; // Colon operator; may have delayed creation.
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor;
    begin
      fGraph := new Graph;
      var create_session := (// Anonymous method to use Dispose pattern.
        method: Tuple of (Success: Boolean, Msg: String, SessionHandle: ^TF_Session);
        begin
          using lStatus := new Status do begin
            using opts := new SessionOptions do begin // Nested
              var sess_handle := TF_NewSession(fGraph.Handle, opts.Handle, lStatus.Handle);
              result := (lStatus.OK, lStatus.Message, sess_handle);
            end;
          end;
        end)();

      if not create_session.Success then begin
        raise new SessionCreateException withMessage(create_session.Msg);
      end;

      inherited constructor withHandle(create_session.SessionHandle)
        OnDispose(aHandle->begin
          using lStatus := new Status do begin
            TF_DeleteSession(aHandle, lStatus.Handle);
          end;
        end);
    end;

    method GetTensorInfo(aOutput: NotNull<Output>; aStatus: Status := nil): String;
    begin
      using lStatus := new Status do begin
        var (success, shp) := fGraph.GetShape(aOutput, lStatus);

        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;

        if success then begin
          var name := String.FromPAnsiChars(TF_OperationName(aOutput.Oper.Handle));
          result := $'Tensor ("{name}: {aOutput.Index}", ' +
                    $'shape={shp.ToString}, '+
                    $'dtype={aOutput.DataType.ToString} )';
        end else begin
          result := '';
        end;
      end;
    end;

    property &Graph: Graph
      read begin
        result := fGraph;
      end;

    property Runner: SessionRunner
      read begin
        if not assigned(fRunner) then begin
          fRunner := new SessionRunner withSession(self);
        end;
        fRunner.Reset;
        result := fRunner;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionOptions = public sealed class(TensorFlowObject<TF_SessionOptions>)
  public
    constructor;
    begin
      var sess_opts_handle := TF_NewSessionOptions();
      inherited constructor withHandle(sess_opts_handle) OnDispose(aHandle->TF_DeleteSessionOptions(aHandle));
    end;

    method SetConfig(aProtoData: NotNull<array of Byte>; aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetConfig(Handle, aProtoData, aProtoData.Length, lStatus.Handle);
        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method SetTarget(aTarget: not nullable String);
    begin
      TF_SetTarget(Handle, aTarget.ToAnsiChars(true));
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionRunnerContext nested in SessionRunner = private class(TensorFlowDisposable)
  private
    fInputs: InputList := new InputList;
    fInputValues: TensorList := new TensorList;
    fOutputs: OutputList := new OutputList;
    fTargets: OperationList := new OperationList;
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fInputs.Dispose;
        fInputValues.Dispose;
        fOutputs.Dispose;
        fTargets.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    property Inputs: InputList
      read begin
        result := fInputs;
      end;

    property Outputs: OutputList
      read begin
        result := fOutputs;
      end;

    property InputValues: TensorList
      read begin
        result := fInputValues;
      end;

    property Targets: OperationList
      read begin
        result := fTargets;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionRunner = public sealed class(TensorFlowDisposable)
  private
    fSession: Session := nil; // Not created by SessionRunner.
    fContext: SessionRunnerContext := new SessionRunnerContext;
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        // fSession is NOT created by Runner, so donnot dispose Runner.
        // fSession.Dispose;
        fContext.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withSession(aSession: NotNull<Session>);
    begin
      fSession := aSession;
    end;

    method AddInput(aInput: NotNull<Output>; aValue: Tensor): SessionRunner;
    begin
      fContext.Inputs.Add(aInput);
      fContext.InputValues.Add(aValue);
      result := self;
    end;

    method Fetch(aOutput: NotNull<Output>): SessionRunner;
    begin
      fContext.Outputs.Add(aOutput);
      result := self;
    end;

    method AddTarget(aTarget: NotNull<Operation>): SessionRunner;
    begin
      fContext.Targets.Add(aTarget);
      result := self;
    end;

    method Reset;
    begin
      fContext.Dispose;
      fContext := new SessionRunnerContext;
    end;

    method Run(aOp: NotNull<Output>; aStatus: Status := nil): Tensor;
    begin
      Reset;
      Fetch(aOp);
      result := Run(aStatus):Item[0]; // May return nil.
    end;

    method Run(aStatus: Status := nil) MetaData(aMetaData: Buffer := nil)
      Options(aOpts: Buffer := nil): TensorList;
    begin
      using lStatus := new Status do begin
        var run_options   := ^TF_Buffer(aOpts: Handle);
        var inputs        := fContext.Inputs.AsTFInputs;
        var input_values  := fContext.InputValues.Handles;
        var ninputs       := fContext.Inputs.Count;
        var outputs       := fContext.Outputs.AsTFOutputs;
        var noutputs      := fContext.Outputs.Count;
        var output_values := new ^TF_Tensor[noutputs];
        var target_opers  := fContext.Targets.Handles;
        var ntargets      := fContext.Targets.Count;
        var run_metadata  := ^TF_Buffer(aMetaData:Handle);

        TF_SessionRun(fSession.Handle, run_options, inputs, input_values,
          ninputs, outputs, output_values, noutputs, target_opers,
          ntargets, run_metadata, lStatus.Handle);

        if lStatus.OK then begin
          result := new TensorList withCapacity(noutputs);
          for I: Integer := 0 to noutputs - 1 do begin
            result.Add(new Tensor withHandle(output_values[I]));
          end;
        end else begin
          writeLn($'SessionRunner.Run failed. Code {ord(lStatus.Code)}. {lStatus.Message}');
          result := nil;
        end;

        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;
  end;

  Environment = public sealed static class
  private
    method CheckOsBitSize; static;
    begin
      var os_bit := sizeOf(NativeUInt);
      if os_bit <> sizeOf(UInt64) then begin
        raise new InvalidOsBitSizeException withDetectedOsBitSize(os_bit * sizeOf(Byte));
      end;
    end;
  public
    constructor;
    begin
      CheckOsBitSize;
    end;

    method GetAllOpList: Buffer;
    begin
      result := new Buffer withHandle(TF_GetAllOpList);
    end;

    method GetAllRegisteredKernels(aStatus: Status := nil): Buffer;
    begin
      using lStatus := new Status do begin
        var buf_handle := TF_GetAllRegisteredKernels(lStatus.Handle);
        result := if lStatus.OK then new Buffer withHandle(buf_handle) else nil;
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetRegisteredKernelsForOp(aOpName: NotNull<String>; aStatus: Status := nil): Buffer;
    begin
      using lStatus := new Status do begin
        var buf_handle := TF_GetRegisteredKernelsForOp(aOpName.ToAnsiChars(true), lStatus.Handle);
        result := if lStatus.OK then new Buffer withHandle(buf_handle) else nil;
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    property Version: String
      read begin
        result := String.FromPAnsiChars(TF_Version);
      end;
  end;

end.