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
  RemObjects.Elements.System, 
  TensorFlow;

type
  DisposableObject = public abstract class(IDisposable)
  protected
    method Dispose(aDisposing: Boolean); virtual; empty;
    
    method CheckAndRaiseOnDisposed(aDisposed: Boolean);
    begin
      if aDisposed then raise new ObjectDisposedException(self);
    end;
  public
    method Dispose;
    begin
      Dispose(true);
      // BoehmGC.SuppressFinalize(self);
    end;
  end;

  ObjectDisposedException = public class(Exception)
  public
    constructor (aObject: DisposableObject);
    begin
      inherited constructor($'{aObject.ToString} instance was already disposed.');
    end;
  end;

  TensorFlowObjectDisposeAction<T> = public block(aObjectPtr: ^T);

  TensorFlowObject<T> = public abstract class(DisposableObject)
  private
    fDisposeAction: TensorFlowObjectDisposeAction<T>;
    fDisposed: Boolean := false;
    fObjectPtr: ^T := nil;
    
    finalizer;
    begin
      if not fDisposed then Dispose(false);
    end;
  protected
    constructor withObjectPtr(aObjectPtr: ^T) DisposeAction(aAction: TensorFlowObjectDisposeAction<T>);
    begin
      fObjectPtr := aObjectPtr;
      fDisposeAction := aAction;
    end;

    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then
        exit;
      
      if aDisposing then begin
        // Derived class should call managed object's Dispose().
      end;

      if assigned(fDisposeAction) then 
        fDisposeAction(fObjectPtr);      
      
      fDisposed := true;
      inherited Dispose(aDisposing);
    end;
  public
    property ObjectPtr: ^T 
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        exit fObjectPtr;
      end;
  end;

  Buffer = public class(TensorFlowObject<TF_Buffer>)
  private
    fData: ^Void := nil;
    fDisposeAction: TensorFlowObjectDisposeAction<TF_Buffer> := aObjectPtr->TF_DeleteBuffer(aObjectPtr); 
    fDisposed: Boolean := false;
    fManaged: Boolean := true; // Wheter buffer managed by this class.
    fNumBytes: UInt64 := 0;

    finalizer;
    begin
      if not fDisposed then Dispose(false);
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then 
        exit;
      
      if (assigned(fData) and fManaged) then 
        free(fData);
      
      fDisposed := true;
      inherited Dispose(aDisposing);
    end;
  public
    constructor withFile(aFile: not nullable String);
    begin
      if not File.Exists(aFile) then 
        raise new Exception($'{aFile} not existing.');      
      
      var fs := new FileStream(aFile, FileMode.Open, FileAccess.Read);
      fNumBytes := fs.Length;      
      var buf: ^TF_Buffer := TF_NewBuffer();
      
      if fNumBytes  > 0 then begin
        fData := malloc(fNumBytes);       
        fs.Read(fData, fNumBytes);
      end; 

      fs.Close;
      
      buf^.data := fData;
      buf^.length := fNumBytes;
      buf^.data_deallocator := nil;
      
      fManaged := true;
      inherited constructor withObjectPtr(buf) DisposeAction(fDisposeAction);
    end;

    constructor withString(const aProtoBuf: not nullable String);
    begin
      var buf: ^TF_Buffer := TF_NewBufferFromString(
        aProtoBuf.ToAnsiChars(true), lstrlenA(aProtoBuf.ToAnsiChars(true)));

      fManaged := false;
      fData := buf^.data;
      fNumBytes := buf^.length;
      inherited constructor withObjectPtr(buf) DisposeAction(fDisposeAction);
    end;

    method ToArray: array of Byte;
    begin
      CheckAndRaiseOnDisposed(fDisposed);
      if fNumBytes > 0 then begin
        result := new Byte[fNumBytes];
        memcpy(result, fData, fNumBytes);
      end else begin
        result := nil;
      end;
    end;

    property NumBytes: UInt64 
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        result:= fNumBytes;
      end;
  end;

  Operation = public class(TensorFlowObject<TF_Operation>)
  private
    fName: not nullable String;
    
    finalizer;
    begin
      Dispose(false);
    end;
  public
    constructor withObjectPtr(aPtr: ^TF_Operation) Name(aName: not nullable String);
    begin
      fName := aName;
      inherited constructor withObjectPtr(aPtr) DisposeAction(nil);
    end;

    method ToString: String; override;
    begin
      result := $'Operation: {Convert.UInt64ToHexString(NativeInt(ObjectPtr), 16)}';
    end;

    property Name: not nullable String read fName;
  end;

  Status = public class(TensorFlowObject<TF_Status>)
  private
    finalizer;
    begin
      Dispose(false);
    end;
  public
    constructor;
    begin
      inherited constructor withObjectPtr(TF_NewStatus()) 
        DisposeAction(aObjectPtr->TF_DeleteStatus(aObjectPtr));
    end;

    method SetStatus(aCode: TF_Code) StatusMessage(const aMsg: String);
    begin
      TF_SetStatus(self.ObjectPtr, aCode, aMsg.ToAnsiChars(true));
    end;

    class method ForwardOrCreate(aIncoming: Status): Status;
    begin
      result := if assigned(aIncoming) then aIncoming else new Status;
    end;

    property OK: Boolean
      read begin
        result := StatusCode = TF_Code.TF_OK;
      end;
    
    property StatusCode: TF_Code 
      read begin
        result := TF_GetCode(ObjectPtr);
      end;
    
    property StatusMessage: String
      read begin
        result.FromPAnsiChars(TF_Message(ObjectPtr));
      end;
  end;

  ScopeRestoreAction = public block(const aScopeToRestore: String);

  Scope = public class(DisposableObject)
  private
    fSavedScope: String;
    fRestoreAction: ScopeRestoreAction;
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then 
        exit;
      
      fRestoreAction(fSavedScope);
      fDisposed := true;
      inherited Dispose(aDisposing);
    end;
  public
    constructor withScopeToSave(const aScope: String) RestoreAction(aAction: ScopeRestoreAction);
    begin
      fSavedScope := aScope;
      fRestoreAction := aAction;
    end;
  end;

  TensorShape = public class(DisposableObject)
  private
    fDims: ^Int64;
    fNumDims: Int32;
    fDisposed: Boolean:= false;
    
    finalizer;
    begin
      if not fDisposed then Dispose(false);
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then 
        exit;      
      
      free(fDims);
      fDisposed:= true;
      inherited Dispose(aDisposing);
    end;
  public
    constructor withDimentions(aDims: array of Int64);
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
      CheckAndRaiseOnDisposed(fDisposed);      
      if assigned(fDims) then begin
        result := new Int64[NumDims];
        memcpy(result, fDims, sizeOf(Int64) * NumDims);
      end else begin
        raise new Exception('Zero dimension cannot be converted to array.');
      end;
    end;
    
    property NumDims: Int32
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        result := fNumDims;
      end;
   
    property Dim[aIndex: Int32]: Int64
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        if (NumDims > 0) and (0 <= aIndex < NumDims) then 
          result := fDims[aIndex] 
        else
          raise new Exception($'Shape.Dims[{aIndex}]: invalid index. Shape.NumDims is {NumDims}');
      end;
  end;

  Output = public class(DisposableObject)
  private
    fOper: Operation;
    fIndex: Integer;

    finalizer;
    begin
      Dispose(false);
    end;
  public
    constructor withOperation(aOper: not nullable Operation) OutputIndex(aIndex: Integer);
    begin
      fOper := aOper;
      fIndex := aIndex;
    end;

    method ToTensorFlowNativeOutput: TF_Output;
    begin
      result.oper := self.Oper.ObjectPtr;
      result.index := self.Index_;
    end;

    method ToString: String; override;
    begin
      result := $'[Output: Operation = {self.Oper.ToString} Index = {self.Index_}]';
    end;

    property Oper: Operation
      read begin 
        result := fOper; 
      end;
    
    property Index_: Integer
      read begin
        result := fIndex;
      end;
    
    property OutputType: TF_DataType
      read begin
        result := TF_OperationOutputType(self.ToTensorFlowNativeOutput);
      end;
  end;

  Graph = public class(TensorFlowObject<TF_Graph>)
  private
    fCurrentScope: String;
    
    finalizer;
    begin
      Dispose(false);
    end;
  public
    constructor;
    begin
      inherited constructor withObjectPtr(TF_NewGraph()) 
        DisposeAction(aObjectPtr->TF_DeleteGraph(aObjectPtr));
    end;

    method WithScope(aNewScope: not nullable String): Scope;
    begin
      result := new Scope withScopeToSave(fCurrentScope) 
        RestoreAction(aScopeToRestore->begin fCurrentScope := aScopeToRestore end);
      
      if String.IsNullOrEmpty(CurrentScope) then 
        fCurrentScope := aNewScope 
      else 
        fCurrentScope := fCurrentScope + '/' + aNewScope;
    end;

    method GetOperationByName(const aName: not nullable String): Tuple of (Boolean, Operation);
    begin
      var opPtr := TF_GraphOperationByName(ObjectPtr, aName.ToAnsiChars(true));
      if assigned(opPtr) then 
        result := (true, new Operation withObjectPtr(opPtr) Name(aName))
      else
        result := (false, nil);
    end;

    method GetTensorShape(aOutput: Output; aStatus: Status := nil): Tuple of (Boolean, TensorShape);
    begin
      result := (false, nil);
 
      using disposableStatus := Status.ForwardOrCreate(aStatus) do begin
        var nativeOut := aOutput.ToTensorFlowNativeOutput;
        
        var numDims := TF_GraphGetTensorNumDims(ObjectPtr, nativeOut, 
          disposableStatus.ObjectPtr);         
        if (not disposableStatus.OK) or (numDims = 0) then
          exit;
        
        var dims := new Int64[numDims];        
        TF_GraphGetTensorShape(ObjectPtr, nativeOut, dims, numDims, 
          disposableStatus.ObjectPtr);        
        if disposableStatus.OK then 
          result := (true, new TensorShape withDimentions(dims));
      end;
    end;

    property CurrentScope: String read fCurrentScope;
  end;

  ITensorData = public interface(IDisposable)
    method ToArray: array of Byte;
    property DataType: TF_DataType read;
    property NumBytes: UInt64 read;
    property Shape: TensorShape read;
  end; 

  TensorData<T> = public class(DisposableObject, ITensorData)
  private
    fNumBytes: UInt64 := 0;
    fData: ^Void;
    fDataType: TF_DataType;
    fDisposed: Boolean := false;
    fShape: TensorShape;

    method ToTensorFlowType(aType: &Type): TF_DataType;
    begin
      case aType.Code of
        TypeCodes.Boolean: 
          result := TF_DataType.TF_BOOL;
        TypeCodes.Byte: 
          result := TF_DataType.TF_UINT8;
        TypeCodes.UInt16: 
          result := TF_DataType.TF_UINT16;
        TypeCodes.UInt32: 
          result := TF_DataType.TF_UINT32;
        TypeCodes.UInt64: 
          result := TF_DataType.TF_UINT64;
        TypeCodes.SByte: 
          result := TF_DataType.TF_INT8;
        TypeCodes.Int16: 
          result := TF_DataType.TF_INT16;
        TypeCodes.Int32: 
          result := TF_DataType.TF_INT32;
        TypeCodes.Int64: 
          result := TF_DataType.TF_INT64; 
        TypeCodes.Single: 
          result := TF_DataType.TF_FLOAT;
        TypeCodes.Double: 
          result := TF_DataType.TF_DOUBLE; 
        TypeCodes.String: 
          result := TF_DataType.TF_STRING;
      else
        raise new Exception($'Cannot convert {aType.ToString} to TensorFlow data type.');
      end;
    end;
      
    finalizer;
    begin
      if not fDisposed then Dispose(false);
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then 
        exit;
      
      if aDisposing then 
        fShape.Dispose;
      
      free(fData);
      fDisposed := true;
      inherited Dispose(aDisposing);
    end;
  public
    constructor withValue(aValue: not nullable array of T) Shape(aShape: not nullable TensorShape);
    begin
      fDataType := ToTensorFlowType(aValue[0].GetType);
      fShape := aShape;

      if fDataType <> TF_DataType.TF_STRING then begin
        fNumBytes := TF_DataTypeSize(fDataType) * aValue.Length;
        fData := malloc(fNumBytes); 
        memcpy(fData, aValue, fNumBytes);
      end else begin
        for I: Integer := 0 to aValue.Length -1 do begin
          fNumBytes := fNumBytes + String(aValue[I]).Length + 1;
        end;

        fData := malloc(fNumBytes);
        var curPos: Integer := 0;
        for I: Integer := 0 to aValue.Length - 1 do begin
          var num := String(aValue[I]).Length + 1; // One extra byte for null terminator.
          memcpy(fData + curPos, String(aValue[I]).ToAnsiChars(true), num);
          curPos := curPos + num;
        end;
      end;
    end;

    method ToArray: array of Byte;
    begin
      CheckAndRaiseOnDisposed(fDisposed);
      result := new Byte[fNumBytes];
      memcpy(result, fData, fNumBytes);
    end;

    property NumBytes: UInt64 
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        result := fNumBytes 
      end;
    
    property DataType: TF_DataType 
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        result := fDataType 
      end;
    
    property Shape: TensorShape 
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        result := fShape 
      end;
  end;

  Tensor = public class(TensorFlowObject<TF_Tensor>)
  private
    fData: ITensorData;
    fDisposed: Boolean := false;
    
    finalizer;
    begin
      if not fDisposed then Dispose(false);
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then 
        exit;
      
      if aDisposing then 
        fData.Dispose;
      
      fDisposed := true;
      inherited Dispose(aDisposing);
    end;
  public
    constructor withData(aData: ITensorData);
    begin
      var lTensor := TF_NewTensor(
        aData.DataType, 
        aData.Shape.ToArray, 
        aData.Shape.NumDims, 
        aData.ToArray, 
        aData.NumBytes, 
        nil, 
        nil);

      if not assigned(lTensor) then 
        raise new Exception('Cannot create new Tensor.');      
      
      fData := aData;
      inherited constructor withObjectPtr(lTensor) 
        DisposeAction(aObjectPtr->TF_DeleteTensor(aObjectPtr));
    end;

    property Data: ITensorData
      read begin
        CheckAndRaiseOnDisposed(fDisposed);
        result := fData;
      end;
  end; 

end.