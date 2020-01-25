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
  TensorFlow;

type
  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Graph_Operations = public extension class(Graph)
  private
    method CreateOpDescription(const aOpType: not nullable String; 
      aOpName: not nullable String): OperationDescription;
    begin 
      result := new OperationDescription withGraph(self) OpType(aOpType) 
        OpName(MakeName(aOpType, aOpName));
    end;
    
    method CreateOpOutput(const aOpType: not nullable String; 
      x: not nullable Output; aOpName: not nullable String): Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInput(x);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;

    method CreateOpOutput(const aOpType: not nullable String; 
      x, y: not nullable Output; aOpName: not nullable String): Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInput(x);
      lOpDesc.AddInput(y);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;

    method CreateOpOutput(const aOpType: not nullable String;
      x, y, z: not nullable Output; aOpName: not nullable String)
      : Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInput(x);
      lOpDesc.AddInput(y);
      lOpDesc.AddInput(z);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;

    method CreateOpOutput(const aOpType: not nullable String; 
      aInputs: not nullable array of Output; aOpName: not nullable String)
      : Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInputs(aInputs);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;
  public
    method OpAbort(const aErrMsg: not nullable String := ''; 
      aNoErrOnExit: Boolean := true; aOpName: not nullable String := '')
      : Operation;
    begin
      const lOpType: String = 'Abort';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
  
      if aErrMsg.Length > 0 then begin
        lOpDesc.SetAttrString('error_msg', aErrMsg);
      end;
  
      lOpDesc.SetAttrBool('exit_without_error', aNoErrOnExit);
      (nil, result) := lOpDesc.FinishOperation;
    end;

    method OpAbs(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Abs';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpAcos(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
     const lOpType: String = 'Acos';
     result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpAdd(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Add';
      result := CreateOpOutput(lOpType, x, y, aOpName);
    end;

    method OpAddN(aInputs: not nullable array of Output; 
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'AddN';
      result := CreateOpOutput(lOpType, aInputs, aOpName);
    end;

    method OpArgMax(aInput, aDimension: not nullable Output; 
      aOpName: not nullable String): Output;
    begin
      const lOpType: String = 'ArgMax';
      result := CreateOpOutput(lOpType, aInput, aDimension, aOpName);
    end;

    method OpArgMin(aInput, aDimension: not nullable Output; 
      aOpName: not nullable String): Output;
    begin
      const lOpType: String = 'ArgMin';
      result := CreateOpOutput(lOpType, aInput, aDimension, aOpName);
    end;

    method OpAssignVariable(aResource, aValue: not nullable Output; 
      aOpName: not nullable String): Output;
    begin
      const lOpType: String = 'AssignVariableOp';
      result := CreateOpOutput(lOpType, aResource, aValue, aOpName);
    end;

    method OpAsin(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Asin';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpAtan(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Atan';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpConst(aValue: not nullable Tensor; aOpName: not nullable String := '')
      : Output; overload;
    begin
      result := OpConst(aValue, aValue.Data.DataType, aOpName);
    end;

    method OpConst(aValue: not nullable Tensor; aDataType: TF_DataType; 
      aOpName: not nullable String := ''): Output; overload;
    begin
      const lOpType: String = 'Const';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
      lOpDesc.SetAttrTensor('value', aValue);
      lOpDesc.SetAttrType('dtype', aDataType);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;

    method OpCos(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Cos';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpDiv(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Div';
      result := CreateOpOutput(lOpType, x, y, aOpName);
    end;

    method OpMatMul(a, b: not nullable Output; transpose_a: Boolean := false;
      transpose_b: Boolean := false; aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'MatMul';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
      lOpDesc.AddInput(a);
      lOpDesc.AddInput(b);
      lOpDesc.SetAttrBool('transpose_a', transpose_a);
      lOpDesc.SetAttrBool('transpose_b', transpose_b);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;

    method OpMul(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Mul';
      result := CreateOpOutput(lOpType, x, y, aOpName);
    end;

    method OpPlaceholder(aDataType: TF_DataType; aShape: Shape := nil; 
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'Placeholder';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
      if assigned(aShape) then lOpDesc.SetAttrShape('shape', aShape);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;    
    end;

    method OnRange(aStart, aLimit, aDelta: not nullable Output; 
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'Range';
      result := CreateOpOutput(lOpType, aStart, aLimit, aDelta, aOpName);
    end;

    method OpReadVariable(aResource: not nullable Output; aDataType: TF_DataType; 
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'ReadVariableOp';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
      lOpDesc.SetAttrType('dtype', aDataType);
      var (success, op) := lOpDesc.FinishOperation;
      result := if success then new Output withOp(op) else nil;
    end;

    method OpSin(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Sin';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpSub(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Sub';
      result := CreateOpOutput(lOpType, x, y, aOpName);
    end;

    method OpTan(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Tan';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;
  end;

end.