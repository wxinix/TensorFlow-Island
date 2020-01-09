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
  TensorFlow,
  RemObjects.Elements.System;

type
  TensorFlowObjectDisposedException<T> = public class(Exception)
  public
    constructor (aObject: TensorFlowObject<T>);
    begin
      inherited constructor($'{aObject.ToString} instance was already disposed.');
    end;
  end;

  ObjectDisposeAction<T> = public block(aObjectPtr: ^T);

  TensorFlowObject<T> = public abstract class(IDisposable)
  private
    fDisposeAction: ObjectDisposeAction<T>;
    fDisposed: Boolean := false;
    fObjectPtr: ^T := nil;

    finalizer;
    begin
      if not fDisposed then Dispose(false);
    end;
  protected
    constructor withObjectPtr(aObjectPtr: ^T) DisposeAction(aAction: ObjectDisposeAction<T>);
    begin
      fObjectPtr := aObjectPtr;
      fDisposeAction := aAction;
    end;

    method Dispose(aDisposing: Boolean); virtual;
    begin
      if fDisposed then exit;     
      if assigned(fDisposeAction) then fDisposeAction(fObjectPtr);
      fDisposed := true;
    end;
  public
    method Dispose;
    begin
      Dispose(true);
    end;

    property ObjectPtr: ^T 
      read begin
        if fDisposed then 
          raise new TensorFlowObjectDisposedException<T>(self)
        else
          exit fObjectPtr;
      end;
  end;

  Buffer = public class(TensorFlowObject<TF_Buffer>)
  private
    fDisposeAction: ObjectDisposeAction<TF_Buffer> := aObjectPtr->TF_DeleteBuffer(aObjectPtr);  
  public
    constructor;
    begin
      inherited constructor withObjectPtr(TF_NewBuffer()) 
        DisposeAction(fDisposeAction);
    end;

    constructor(const aProtoBuf: array of Byte);
    begin
      inherited constructor withObjectPtr(TF_NewBufferFromString(aProtoBuf, aProtoBuf.Length))
        DisposeAction(fDisposeAction);
    end;
  end;

  Operation = public class(TensorFlowObject<TF_Operation>)
  private
    fName: not nullable String;
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

  Scope = public class(IDisposable)
  private
    fSavedScope: String;
    fRestoreAction: ScopeRestoreAction;
  public
    constructor withScopeToSave(const aScope: String) RestoreAction(aAction: ScopeRestoreAction);
    begin
      fSavedScope := aScope;
      fRestoreAction := aAction;
    end;

    method Dispose;
    begin
      fRestoreAction(fSavedScope);
    end;
  end;

  Shape = public class
  private
    fDims: array of int64_t;
  public
    constructor withDimenstions(aDims: array of Int64);
    begin
      fDims := aDims;
    end;

    method ToArray: array of Int64;
    begin
      exit fDims;
    end;

    property NumDims: Int32 read fDims.Length;
    property Dims[aIndex: Int32]: Int64
      read begin
        result := if (NumDims > 0) and (0 <= aIndex < NumDims) then 
                    fDims[aIndex] 
                  else
                    raise new Exception($'Shape.Dims[{aIndex}]: invalid index. Shape.NumDims is {NumDims}');
      end;
  end;

  Output = public class
  private
    fOper: Operation;
    fIndex: Integer;
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
      
      fCurrentScope := if String.IsNullOrEmpty(CurrentScope) then 
                         aNewScope 
                       else 
                         fCurrentScope + '/' + aNewScope;
    end;

    method GetOperationByName(const aName: not nullable String): Tuple of (Boolean, Operation);
    begin      
      var opPtr := TF_GraphOperationByName(ObjectPtr, aName.ToAnsiChars(true));
      if assigned(opPtr) then 
        result := (true, new Operation withObjectPtr(opPtr) Name(aName))
      else
        result := (false, nil);
    end;

    method GetTensorShape(aOutput: Output; aStatus: Status := nil): Tuple of (Boolean, Shape);
    begin
      result := (false, nil);
      var nativeOut := aOutput.ToTensorFlowNativeOutput;

      using disposableStatus := Status.ForwardOrCreate(aStatus) do 
      begin
        var numDims := TF_GraphGetTensorNumDims(ObjectPtr, nativeOut, disposableStatus.ObjectPtr);
        if (not disposableStatus.OK) or (numDims = 0) then 
          exit;
        var dims := new Int64[numDims];
        TF_GraphGetTensorShape(ObjectPtr, nativeOut, dims, numDims, disposableStatus.ObjectPtr);
        if disposableStatus.OK then
          result := (true, new Shape withDimenstions(dims));
      end;
    end;

    property CurrentScope: String read fCurrentScope;
  end;

end.