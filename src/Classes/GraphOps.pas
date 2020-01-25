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
  OpCreateException = public class(Exception)
  public
    constructor withOpType(aOpType: not nullable String) 
      Message(aMsg: not nullable String := '');
    begin
      inherited constructor($'Fail creating {aOpType}. {aMsg}');
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Graph_Operations = public extension class(Graph)
  private
    method CreateOpDescription(const aOpType: not nullable String; 
      aOpName: not nullable String): OperationDescription;
    begin 
      result := new OperationDescription withGraph(self) OpType(aOpType) 
        OpName(MakeName(aOpType, aOpName));
    end;

    method FinishOpDescription_Output(aOpDesc: not nullable OperationDescription)
      : Output; 
    begin
      using lStatus := new Status do begin
        var (success, op) := aOpDesc.FinishOperation(lStatus);
        if success then 
          result:= new Output withOp(op) 
        else 
          raise new OpCreateException withOpType(aOpDesc.OpType) 
            Message(lStatus.Message);
      end;
    end;

    method FinishOpDescription_Op(aOpDesc: not nullable OperationDescription)
      : Operation;
    begin
      using lStatus := new Status do begin
        var (success, op) := aOpDesc.FinishOperation(lStatus);
        if success then begin
          result := op;
        end else begin
          raise new OpCreateException withOpType(aOpDesc.OpType)
            Message(lStatus.Message);
        end;
      end;
    end;
    
    method CreateOpOutput(const aOpType: not nullable String; 
      x: not nullable Output; aOpName: not nullable String): Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInput(x);
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method CreateOpOutput(const aOpType: not nullable String; 
      x, y: not nullable Output; aOpName: not nullable String): Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInput(x);
      lOpDesc.AddInput(y);
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method CreateOpOutput(const aOpType: not nullable String;
      x, y, z: not nullable Output; aOpName: not nullable String := '')
      : Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInput(x);
      lOpDesc.AddInput(y);
      lOpDesc.AddInput(z);
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method CreateOpOutput(const aOpType: not nullable String; 
      aInputs: not nullable array of Output; aOpName: not nullable String)
      : Output; overload;
    begin
      var lOpDesc := CreateOpDescription(aOpType, aOpName);
      lOpDesc.AddInputs(aInputs);
      result := FinishOpDescription_Output(lOpDesc);
    end;
  public
    method OpAbort(const aErrMsg: not nullable String := ''; 
      aNoErrOnExit: Boolean := true; aOpName: not nullable String := '')
      : Operation;
    begin
      const lOpType: String = 'Abort';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
  
      if aErrMsg.Length > 0 then lOpDesc.SetAttrStr('error_msg', aErrMsg);
      lOpDesc.SetAttrBool('exit_without_error', aNoErrOnExit);
      
      result := FinishOpDescription_Op(lOpDesc);
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
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'ArgMax';
      result := CreateOpOutput(lOpType, aInput, aDimension, aOpName);
    end;

    method OpArgMin(aInput, aDimension: not nullable Output; 
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'ArgMin';
      result := CreateOpOutput(lOpType, aInput, aDimension, aOpName);
    end;

    method OpAssignVariable(aResource, aValue: not nullable Output; 
      aOpName: not nullable String := ''): Operation;
    begin
      const lOpType: String = 'AssignVariableOp';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
      lOpDesc.AddInput(aResource);
      lOpDesc.AddInput(aValue);
      result := FinishOpDescription_Op(lOpDesc);
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
      result := FinishOpDescription_Output(lOpDesc);
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
      result := FinishOpDescription_Output(lOpDesc);
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
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method OpRange(aStart, aLimit, aDelta: not nullable Output; 
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
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method OpReduceDims(aInput: not nullable Output; aAxis: Output := nil): Output;
    begin
      if assigned(aAxis) then begin
        result := aAxis;
      end else begin
        using lStatus := new Status do begin
          var (success, shp) := self.GetShape (aInput, lStatus);
          if success then begin
            if shp.NumDims > 0 then begin
              var arr := new Int32[shp.NumDims];
              for I: Integer := 0 to shp.NumDims - 1 do arr[I] := I;
              result := OpConst(arr, TF_DataType.TF_INT32);
            end else begin
              result := OpRange(OpConst(0), OpConst(0), OpConst(1));
            end;
          end else begin
            raise new OpCreateException withOpType('OpReduceDims') 
              Message(lStatus.Message)
          end;
        end;
      end;
    end;

    method OpReduceSum(aInput, aAxis: not nullable Output; 
      aKeepDims: Boolean := false; aOpName: not nullable String := ''): Output;
    begin
      var reductionIndices := OpReduceDims(aInput, aAxis);
      result := OpSum(aInput, reductionIndices, aKeepDims, aOpName);
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

    method OpSum(aInput, aReductionIndices: not nullable Output; 
      aKeepDims: Boolean := false; aOpName: not nullable String := ''): Output;
    begin
        const lOpType: String = 'Sum';
        var lOpDesc := CreateOpDescription(lOpType, aOpName);
        lOpDesc.AddInput(aInput);
        lOpDesc.AddInput(aReductionIndices);
        lOpDesc.SetAttrBool('keep_dims', aKeepDims);
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method OpTan(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Tan';
      result := CreateOpOutput(lOpType, x, aOpName);
    end;

    method OpVarHandle(aDataType: TF_DataType; aShape: not nullable Shape;
      aContainer: not nullable String := ''; aSharedName: not nullable String := '';
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'VarHandleOp';
      var lOpDesc := CreateOpDescription(lOpType, aOpName);
      lOpDesc.SetAttrType('dtype', aDataType);
      lOpDesc.SetAttrShape('shape', aShape);
      
      if aContainer.Length  > 0 then lOpDesc.SetAttrStr('container', aContainer);
      if aSharedName.Length > 0 then lOpDesc.SetAttrStr('shared_name', aSharedName);
      
      result := FinishOpDescription_Output(lOpDesc);
    end;

    method Variable(aInitialValue: not nullable Output; aOpName: not nullable String := '')
      : Tuple of (Operation, Output, Output); //(OpAssignVar, OpReadVar, OpVarHandle)
    begin
      var opAssignVar: Operation;
      var opReadVar, opVarHnd: Output;

      using variableScope := WithScope(MakeName('Variable', aOpName)) do begin
        using lStatus := new Status do begin
          var (success, shp) := GetShape(aInitialValue, lStatus);          
          
          if not success then begin
            raise new OpCreateException withOpType('Variable') 
              Message(lStatus.Message);
          end;
          
          using shp do begin 
            opVarHnd := OpVarHandle(aInitialValue.Type, shp);
          end;

          using assignScope := WithScope('Assign') do begin
            opAssignVar := OpAssignVariable(opVarHnd, aInitialValue);
            using readScope := WithScope('Read') do begin
              opReadVar := OpReadVariable(opVarHnd, aInitialValue.Type);
            end;
          end;
        end;
      end;

      result := (opAssignVar, opReadVar, opVarHnd);
    end;
  end;
end.