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
  TensorFlow.Island.Aspects,
  RemObjects.Elements.System,
  TensorFlow;

type
  NotNull<T> = public not nullable T;

  DisposableObject = public abstract class(IDisposable)
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
  public
    method Dispose;
    begin
      if not fDisposed then begin
        Dispose(true);
        BoehmGC.SuppressFinalize(self);
      end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  DisposableObjectList<T> = public abstract class(DisposableObject, IEnumerable<T>)
    where T is IDisposable;
  protected
    fList: List<T>; implements IEnumerable<T>;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if aDisposing then begin
        for el in fList do el.Dispose();
      end;

      fList.Clear;
      inherited Dispose(aDisposing);
    end;
  public
    constructor withCapacity(aCapacity: Integer);
    begin
      fList := new List<T>(aCapacity);
    end;

    constructor;
    begin
      fList := new List<T>;
    end;

    method &Add(aItem: T);
    begin
      fList.Add(aItem);
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

  TensorFlowObjectDisposeAction<T> = public block(aPtr: ^T);

  ITensorFlowObject = unit interface(IDisposable) // unit visibility.
    property ID: NativeUInt read;
    property RawPtr: ^Void read;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowObject<T> = public abstract class(DisposableObject, ITensorFlowObject)
  private
    fDisposeAction: TensorFlowObjectDisposeAction<T>;
    fPtr: ^T := nil;
    fDisposed: Boolean := false;
  protected
    constructor withPtr(aPtr: NotNull<^T>) DisposeAction(aAction: TensorFlowObjectDisposeAction<T>);
    begin
      fPtr := aPtr;
      fDisposeAction := aAction;
    end;

    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end;

      if aDisposing then begin
        // Derived class should call its managed object's Dispose().
      end;

      if assigned(fDisposeAction) then begin
        fDisposeAction(fPtr);
      end;

      fDisposed := true;
      inherited Dispose(aDisposing);
    end;
  public
    property ID: NativeUInt
      read begin
        result := NativeInt(fPtr);
      end;

    property Ptr: ^T // Typed pointer.
      read begin
        exit fPtr;
      end;

    property RawPtr: ^Void // Untyped pointer.
      read begin
        result := fPtr;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowObjectList<T> = public class(DisposableObjectList<T>)
    where T is ITensorFlowObject;
  public
    method ToRawPtrArray: array of ^Void;
    begin
      if fList.Count = 0 then exit nil;
      result := new ^Void[fList.Count];
      for I: Integer := 0 to fList.Count - 1 do begin
        result[I] := fList[I].RawPtr;
      end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Buffer = public class(TensorFlowObject<TF_Buffer>)
  private
    fData: ^Void := nil;
    fDisposeAction: TensorFlowObjectDisposeAction<TF_Buffer> := aPtr->TF_DeleteBuffer(aPtr);
    fManaged: Boolean := true; // Whether buffer managed by this class.
    fNumBytes: UInt64 := 0;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if (assigned(fData) and fManaged) then begin
        free(fData);
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withFile(aFile: NotNull<String>);
    begin
      var bufData := Helper.ReadBufferDataFromFile(aFile);

      if assigned(bufData) then begin
        fNumBytes := bufData.Length;
        fData := malloc(fNumBytes);
        memcpy(fData, bufData, fNumBytes);
      end else begin
        fNumBytes := 0;
        fData := nil;
      end;

      var buf := TF_NewBuffer();
      buf^.data := fData;
      buf^.length := fNumBytes;
      buf^.data_deallocator := nil;

      fManaged := true;
      inherited constructor withPtr(buf) DisposeAction(fDisposeAction);
    end;

    constructor withString(const aProtoBuf: NotNull<String>);
    begin
      var proto_len := lstrlenA(aProtoBuf.ToAnsiChars(true));
      var buf: ^TF_Buffer := TF_NewBufferFromString(aProtoBuf.ToAnsiChars, proto_len);

      fManaged := false;
      fData := buf^.data;
      fNumBytes := buf^.length;
      inherited constructor withPtr(buf) DisposeAction(fDisposeAction);
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
  Operation = public class(TensorFlowObject<TF_Operation>)
  private
    fName: String;
    fGraph: Graph;
  public
    constructor withPtr(aPtr: ^TF_Operation) Name(aName: NotNull<String>) Graph(aGraph: NotNull<Graph>);
    begin
      fGraph := aGraph;
      fName := aName;
      inherited constructor withPtr(aPtr) DisposeAction(nil);
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

  OperationList = public TensorFlowObjectList<Operation>;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  OperationDescription = public class(TensorFlowObject<TF_OperationDescription>)
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

      var opDesc := TF_NewOperation(aGraph.Ptr, aType.ToAnsiChars(true), aName.ToAnsiChars(true));
      // DisposeAction nil, TF_FinishOption will delete OperationDescription.
      inherited constructor withPtr(opDesc) DisposeAction(nil);
    end;

    method SetDevice(aDevice: not nullable String);
    begin
      TF_SetDevice(Ptr, aDevice.ToAnsiChars(true));
    end;

    method AddInput(aInput: NotNull<Output>);
    begin
      TF_AddInput(Ptr, aInput.ToTFOutput);
    end;

    method AddInputs(aInputList: NotNull<array of Output>);
    begin
      var tfOutput := new TF_Output[aInputList.Length];
      for I: Integer := 0 to aInputList.Length - 1 do begin
        tfOutput[I] := aInputList[I].ToTFOutput;
      end;

      TF_AddInputList(Ptr, tfOutput, tfOutput.Length);
    end;

    method FinishOperation(aStatus: Status := nil): Tuple of (Boolean, Operation);
    begin
      using lStatus := new Status do begin
        // Desc ptr gets deleted inside TF_FinishOperation.
        var op := TF_FinishOperation(Ptr, lStatus.Ptr);

        if lStatus.OK then begin
          result := (true, new Operation withPtr(op) Name(fOperName) Graph(fGraph))
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
      TF_SetAttrBool(Ptr, aName.ToAnsiChars(true), value);
    end;

    method SetAttrBoolList(const aName: NotNull<String>; aList: NotNull<array of Boolean>);
    begin
      var values := new Byte[aList.Length];
      for I: Integer := 0 to aList.Length - 1 do begin
        values[I] := if aList[I] then 1 else 0;
      end;

      TF_SetAttrBoolList(Ptr, aName.ToAnsiChars(true), values, values.Length);
    end;

    method SetAttrFloat(const aName: not nullable String; aValue: Single);
    begin
      TF_SetAttrFloat(Ptr, aName.ToAnsiChars(true), aValue);
    end;

    method SetAttrFloatList(const aName: NotNull<String>; aList: NotNull<array of Single>);
    begin
      TF_SetAttrFloatList(Ptr, aName.ToAnsiChars(true), aList, aList.Length);
    end;

    method SetAttrInt(const aName: NotNull<String>; aValue: Int64);
    begin
      TF_SetAttrInt(Ptr, aName.ToAnsiChars(true), aValue);
    end;

    method SetAttrIntList(const aName: NotNull<String>; aList: NotNull<array of Int64>);
    begin
      TF_SetAttrIntList(Ptr, aName.ToAnsiChars(true), aList, aList.Length);
    end;

    method SetAttrStr(const aName: NotNull<String>; aValue: NotNull<String>);
    begin
      var length := lstrlenA(aValue.ToAnsiChars(true));
      TF_SetAttrString(Ptr, aName.ToAnsiChars(true), aValue.ToAnsiChars, length);
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

      TF_SetAttrStringList(Ptr, aName.ToAnsiChars(true), ^^Void(values),
        lengths, num_values);
    end;

    method SetAttrType(const aName: NotNull<String>; aType: TF_DataType);
    begin
      TF_SetAttrType(Ptr, aName.ToAnsiChars(true), aType);
    end;

    method SetAttrTypeList(const aName: NotNull<String>; aTypeList: NotNull<array of TF_DataType>);
    begin
      TF_SetAttrTypeList(Ptr, aName.ToAnsiChars(true), aTypeList, aTypeList.Length);
    end;

    method SetAttrTensor(const aName: NotNull<String>; aTensor: NotNull<Tensor>; 
      aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrTensor(Ptr, aName.ToAnsiChars(true), aTensor.Ptr, lStatus.Ptr);
        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method SetAttrShape(const aName: NotNull<String>; aShape: NotNull<Shape>);
    begin
      TF_SetAttrShape(Ptr, aName.ToAnsiChars(true), aShape.ToArray, aShape.NumDims);
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

      TF_SetAttrShapeList(Ptr, aName.ToAnsiChars(true), ^^Int64(dims), num_dims, num_shapes);
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
  Status = public class(TensorFlowObject<TF_Status>)
  public
    constructor;
    begin
      inherited constructor withPtr(TF_NewStatus()) DisposeAction(aPtr->TF_DeleteStatus(aPtr));
    end;

    method SetCode(aCode: TF_Code) withMessage(const aMsg: String);
    begin
      if not String.IsNullOrEmpty(aMsg) then
        TF_SetStatus(Ptr, aCode, aMsg.ToAnsiChars(true))
      else
        TF_SetStatus(Ptr, aCode, nil);
    end;

    class method ForwardOrCreate(aIncoming: Status): Status;
    begin
      result := if assigned(aIncoming) then aIncoming else new Status;
    end;

    property OK: Boolean
      read begin
        result := Code = TF_Code.TF_OK;
      end;

    property Code: TF_Code
      read begin
        result := TF_GetCode(Ptr);
      end;

    property Message: String
      read begin
        result := String.FromPAnsiChars(TF_Message(Ptr));
      end;
  end;

  ScopeRestoreAction = public block(const aScopeToRestore: NotNull<String>);

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Scope = public class(DisposableObject)
  private
    fRestoreAction: ScopeRestoreAction;
    fSavedScope: String;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      fRestoreAction(fSavedScope);
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
  Shape = public class(DisposableObject)
  private
    fDims: ^Int64;
    fNumDims: Int32;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      free(fDims);
      inherited Dispose(aDisposing);
    end;
  public
    constructor withDims(aDims: array of Int64); // aDims can be nil.
    begin
      fNumDims := if assigned(aDims) then aDims.Length else 0;
      var numBytes := sizeOf(int64_t) * fNumDims;

      if numBytes >0 then begin
        fDims := ^Int64(malloc(numBytes));
        memcpy(fDims, aDims, numBytes);
      end else begin
        fDims := nil;
      end;
    end;

    method ToArray: array of Int64;
    begin
      if assigned(fDims) then begin
        result := new Int64[NumDims];
        memcpy(result, fDims, sizeOf(Int64) * NumDims);
      end else begin
        result := nil;
      end;
    end;

    method ToString: String; override;
    begin
      var dims_str: String := '';
      for I: Integer := 0 to NumDims - 1 do begin
        dims_str :=
          if (I < NumDims - 1) then
            dims_str + $'Dimension({Dim[I]}), '
          else
            dims_str + $'Dimension({Dim[I]})';
      end;
      result := $'TensorShape([{dims_str}])';
    end;

    operator Implicit(aDims: array of Int64): Shape;
    begin
      result := new Shape withDims(aDims);
    end;

    property NumDims: Int32
      read begin
        result := fNumDims;
      end;

    property Dim[aIndex: Int32]: Int64
      read begin
        if (NumDims > 0) and (0 <= aIndex < NumDims) then begin
          result := fDims[aIndex]
        end else begin
          raise new InvalidShapeDimIndexException(aIndex, NumDims);
        end;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Output = public class(DisposableObject)
  private
    fIndex: Integer;
    fOper: Operation;
  public
    constructor withOp(aOp: NotNull<Operation>) Index(aIndex: Integer = 0);
    begin
      fIndex := aIndex;
      fOper := aOp;
    end;

    method ToTFOutput: TF_Output;
    begin
      result.oper  := self.Oper.Ptr;
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
        result := TF_OperationOutputNumConsumers(ToTFOutput());
      end;

    property Oper: Operation
      read begin
        result := fOper;
      end;

    property DataType: TF_DataType
      read begin
        result := TF_OperationOutputType(ToTFOutput);
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  OutputList = public class(DisposableObjectList<Output>)
  public
    method ToTFOutputArray: array of TF_Output;
    begin
      if fList.Count = 0 then exit nil;
      result := new TF_Output[fList.Count];
      for I: Integer := 0 to fList.Count - 1 do begin
        result[I] := fList[I].ToTFOutput;
      end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Graph = public class(TensorFlowObject<TF_Graph>)
  private
    fCurrentScope: not nullable String := '';
    fNamesCache: Dictionary<String, Integer> := new Dictionary<String, Integer>;
    fPendingInitVars: OperationList := new OperationList;

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
      if aDisposing then begin
        fPendingInitVars.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor;
    begin
      var lgraph := TF_NewGraph();
      inherited constructor withPtr(lgraph) DisposeAction(aPtr->TF_DeleteGraph(aPtr));
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
      var op := TF_GraphOperationByName(Ptr, aOpName.ToAnsiChars(true));

      if assigned(op) then begin
        result := (true,
          new Operation withPtr(op) Name(aOpName) Graph(self));
      end else begin
        result := (false, nil);
      end;
    end;

    method GetShape(aOutput: NotNull<Output>; aStatus: Status := nil): Tuple of (Boolean, Shape);
    begin
      using lStatus := new Status do begin
        var nativeOut := aOutput.ToTFOutput;
        var numDims := TF_GraphGetTensorNumDims(Ptr, nativeOut, lStatus.Ptr);

        if (not lStatus.OK) then begin
          result := (false, nil);
        end else begin
          if numDims > 0 then begin
            var dims := new Int64[numDims];
            TF_GraphGetTensorShape(Ptr, nativeOut, dims, numDims, lStatus.Ptr);
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

    method PrintTensorInfo(aOutput: NotNull<Output>; aStatus: Status := nil): String;
    begin
      using lStatus := new Status do begin
        var success: Boolean;
        var shp: Shape;
        (success, shp) := GetShape(aOutput, lStatus);

        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;

        if success then begin
          var name := String.FromPAnsiChars(TF_OperationName(aOutput.Oper.Ptr));
          result := $'Tensor ("{name}: {aOutput.Index}", ' +
                    $'shape={shp.ToString}, '+
                    $'dtype={Helper.TFDataTypeToString(aOutput.DataType)} )';
        end else begin
          result := '';
        end;
      end;
    end;

    property CurrentScope: String
      read begin
        result := fCurrentScope;
      end;

    property GlobalVariableInitializer: array of ^TF_Operation
      read begin
        result := fPendingInitVars.ToRawPtrArray;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorData = public class(DisposableObject)
  protected
    fBytes: ^Void;
    fDataType: TF_DataType;
    fManaged: Boolean;
    fNumBytes: UInt64;
    fShape: Shape;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
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
      // using constructor withTensor, then the associated data will be managed
      // by whoever creates that raw Tensor pointer.
    end;

    constructor withTFTensor(aTensor: NotNull<^TF_Tensor>);
    begin
      fBytes := TF_TensorData(aTensor);
      fDataType := TF_TensorType(aTensor);
      fNumBytes := TF_TensorByteSize(aTensor);

      var lNumDims := TF_NumDims(aTensor);
      var lDims: array of Int64;
      if lNumDims > 0 then begin
        lDims := new Int64[lNumDims];
        for I: Integer := 0 to lNumDims - 1 do begin
          lDims[I] := TF_Dim(aTensor, I);
        end;
      end else begin
        lDims := nil;
      end;

      fManaged := false;
      fShape := new Shape withDims(lDims);
    end;

    method Bytes: ^Void;
    begin
      result := fBytes;
    end;

    property NumBytes: UInt64
      read begin
        result := fNumBytes
      end;

    property DataType: TF_DataType
      read begin
        result := fDataType
      end;

    property Shape: Shape
      read begin
        result := fShape
      end;
  end;

  TensorData<T> = public class(TensorData)
  public
    constructor withValues(aVals: NotNull<array of T>) Shape(aShp: NotNull<Shape>);
    begin
      fDataType := Helper.ToTFDataType(typeOf(T));
      fShape := aShp;
      fManaged := true;

      if fDataType <> TF_DataType.TF_STRING then begin
        fNumBytes := TF_DataTypeSize(fDataType) * aVals.Length;
        fBytes := malloc(fNumBytes);
        if aVals.Length = 1 then begin
          (^T(fBytes))^ := aVals[0];
        end else begin
          memcpy(fBytes, aVals, fNumBytes);
        end;
      end else begin
        for I: Integer := 0 to aVals.Length -1 do begin
          fNumBytes := fNumBytes + String(aVals[I]).Length + 1;
        end;

        fBytes := malloc(fNumBytes);
        var curPos: Integer := 0;

        for I: Integer := 0 to aVals.Length - 1 do begin
          var num := String(aVals[I]).Length + 1; //1 byte for null terminator.
          memcpy(fBytes + curPos, String(aVals[I]).ToAnsiChars(true), num);
          curPos := curPos + num;
        end;
      end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Tensor = public class(TensorFlowObject<TF_Tensor>)
  private
    fData: TensorData;

    class method ConvertToTensor<T>(aVals: NotNull<array of T>): Tensor; overload;
    begin
      var shp := new Shape withDims([aVals.Length]);
      var data := new TensorData<T> withValues(aVals) Shape(shp);
      result := new Tensor withData(data);
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
        memcpy(@arr[I + I * width], @aVals[I][0], width * sizeOf(T));
      end;

      var shp := new Shape withDims([height, width]);
      var data := new TensorData<T> withValues(arr) Shape(shp);
      result := new Tensor withData(data);
    end;

    class method ConvertToTensor<T>(aValue: T): Tensor; overload;
    begin
      var shp := new Shape withDims(nil);
      var data := new TensorData<T> withValues([aValue]) Shape(shp);
      result := new Tensor withData(data);
    end;

  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if aDisposing then begin
        fData.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withData(aData: NotNull<TensorData>);
    begin
      var lTensor := TF_NewTensor(
        aData.DataType,
        aData.Shape.ToArray,
        aData.Shape.NumDims,
        aData.Bytes, // Must be raw bytes; cannot be managed array.
        aData.NumBytes,
        @TensorData.DeallocateTensorData, // does nothing.
        nil);

      if not assigned(lTensor) then begin
        raise new TensorCreateException(aData.DataType);
      end;

      fData := aData;
      inherited constructor withPtr(lTensor) DisposeAction(aPtr->TF_DeleteTensor(aPtr));
    end;

    constructor withTFTensor(aTensor: not nullable ^TF_Tensor);
    begin
      var lData: TensorData := new TensorData withTFTensor(aTensor);
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

    method AsScalar<T>: Tuple of (Boolean, nullable T);
    begin
      if (not IsScalar) or
         (fData.DataType <> Helper.ToTFDataType(typeOf(T)) RaiseOnInvalid(false)) or
         (typeOf(T) = typeOf(String))
      then begin
        result := (false, nil);
      end else begin
        var value: T := (^T(fData.Bytes))^;
        result := (true, value);
      end;
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

  TensorList = public TensorFlowObjectList<Tensor>;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Session = public class(TensorFlowObject<TF_Session>)
  private
    fGraph: Graph;
    fRunner: SessionRunner;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if aDisposing then begin
        fGraph :Dispose; // Colon operator
        fRunner:Dispose; // Colon operator
      end;
      inherited Dispose(aDisposing);
    end;
  public
    constructor;
    begin
      fGraph := new Graph;
      var createSessionResult := (// Anonymous method to use Dispose pattern.
        method: Tuple of (Success: Boolean, Msg: String, SessionPtr: ^TF_Session);
        begin
          using lStatus := new Status do begin
            using opts := new SessionOptions do begin // Nested
              var lSession := TF_NewSession(fGraph.Ptr, opts.Ptr, lStatus.Ptr);
              result := (lStatus.OK, lStatus.Message, lSession);
            end;
          end;
        end
       )();

      if not createSessionResult.Success then begin
        raise new SessionCreateException withMessage(createSessionResult.Msg);
      end;

      inherited constructor withPtr(createSessionResult.SessionPtr)
        DisposeAction(aPtr->begin
          using lStatus := new Status do begin
            TF_DeleteSession(aPtr, lStatus.Ptr);
          end;
        end);
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
  SessionOptions = public class(TensorFlowObject<TF_SessionOptions>)
  public
    constructor;
    begin
      var session_opts := TF_NewSessionOptions();
      inherited constructor withPtr(session_opts) DisposeAction(aPtr->TF_DeleteSessionOptions(aPtr));
    end;

    method SetConfig(aProtoData: NotNull<array of Byte>; aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetConfig(Ptr, aProtoData, aProtoData.Length, lStatus.Ptr);
        if assigned(aStatus) then begin
          aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method SetTarget(aTarget: not nullable String);
    begin
      TF_SetTarget(Ptr, aTarget.ToAnsiChars(true));
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionRunnerContext nested in SessionRunner = private class(DisposableObject)
  private
    fInputs: OutputList := new OutputList;
    fInputValues: TensorList := new TensorList;
    fOutputs: OutputList := new OutputList;
    fTargets: OperationList := new OperationList;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if aDisposing then begin
        fInputs.Dispose;
        fInputValues.Dispose;
        fOutputs.Dispose;
        fTargets.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    property Inputs: OutputList
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
  SessionRunner = public class(DisposableObject)
  private
    fSession: Session := nil;
    fContext: SessionRunnerContext := new SessionRunnerContext;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if aDisposing then begin
        // fSession is NOT created by Runner, so Runner cannot dispose.
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

    method Run(aStatus: Status := nil) MetaData(aMetaData: Buffer := nil) Options(aOpts: Buffer := nil)
      : TensorList;
    begin
      using lStatus := new Status do begin
        var run_options := aOpts: Ptr;
        var inputs := fContext.Inputs.ToTFOutputArray;
        var input_values := fContext.InputValues.ToRawPtrArray;
        var ninputs := fContext.Inputs.Count;
        var outputs := fContext.Outputs.ToTFOutputArray;
        var noutputs := fContext.Outputs.Count;
        var output_values: array of ^TF_Tensor := new ^TF_Tensor[noutputs];
        var target_opers := fContext.Targets.ToRawPtrArray;
        var ntargets := fContext.Targets.Count;
        var run_metadata := aMetaData:Ptr;

        TF_SessionRun(fSession.Ptr, run_options, inputs, input_values,
          ninputs, outputs, output_values, noutputs, target_opers,
          ntargets, run_metadata, lStatus.Ptr);

        if lStatus.OK then begin
          result := new TensorList withCapacity(noutputs);
          for I: Integer := 0 to noutputs - 1 do begin
            result.Add(new Tensor withTFTensor(output_values[I]));
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

end.