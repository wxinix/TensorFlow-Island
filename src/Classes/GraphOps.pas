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
    method InternalCreateOp(const aOpType: not nullable String; aOpName: not nullable String;
      aInputs: array of not nullable Output = [];
      aAttrs: array of Tuple of (not nullable String, not nullable Object) = [])
      : Tuple of (Operation, Output);
    begin
      var lOpDesc := new OperationDescription withGraph(self) OpType(aOpType)
        OpName(MakeName(aOpType, aOpName));

      for each x in aInputs do lOpDesc.AddInput(x);
      for each a in aAttrs  do lOpDesc.SetAttr(a[0], a[1]);

      using lStatus := new Status do begin
        var (success, result_op) := lOpDesc.FinishOperation(lStatus);
        if success then begin
          var result_output := new Output withOp(result_op);
          result := (result_op, result_output);
        end else begin
          raise new OpCreateException withOpType(aOpType) Message(lStatus.Message);
        end;
      end;
    end;
  public
    method OpAbort(const aErrMsg: not nullable String := '';
      aNoErrOnExit: Boolean := true; aOpName: not nullable String := '')
      : Operation;
    begin
      const lOpType: String = 'Abort';

      (result, nil) := InternalCreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('error_msg', aErrMsg),
          ('exit_without_error', aNoErrOnExit)
        ]);
    end;

   method OpAbs(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Abs';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
    end;

    method OpAcos(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
     const lOpType: String = 'Acos';
     (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
    end;

    method OpAdd(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Add';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpAddN(aInputs: not nullable array of Output;
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'AddN';
      (nil, result) := InternalCreateOp(lOpType, aOpName, aInputs);
    end;

    method OpArgMax(aInput, aDimension: not nullable Output;
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'ArgMax';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [aInput, aDimension]);
    end;


    method OpArgMin(aInput, aDimension: not nullable Output;
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'ArgMin';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [aInput, aDimension]);
    end;

    method OpAssignVariable(aResource, aValue: not nullable Output;
      aOpName: not nullable String := ''): Operation;
    begin
      const lOpType: String = 'AssignVariableOp';
      (result, nil) := InternalCreateOp(lOpType, aOpName, [aResource, aValue]);
    end;

    method OpAsin(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Asin';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
    end;

    method OpAtan(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Atan';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
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

      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('value', aValue),
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpCos(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Cos';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
    end;

    method OpDiv(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Div';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpMatMul(a, b: not nullable Output; transpose_a: Boolean := false;
      transpose_b: Boolean := false; aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'MatMul';
      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [a, b],
        [
          ('transpose_a', transpose_a),
          ('transpose_b', transpose_b)
        ]);
    end;

    method OpMul(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Mul';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpPlaceholder(aDataType: TF_DataType; aOpName: not nullable String := '')
      : Output; overload;
    begin
      const lOpType: String = 'Placeholder';
      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpPlaceholder(aDataType: TF_DataType; aShape: not nullable Shape;
      aOpName: not nullable String := ''): Output; overload;
    begin
      const lOpType: String = 'Placeholder';
      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('shape', aShape),
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpRange(aStart, aLimit, aDelta: not nullable Output;
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'Range';
      (nil, result) := InternalCreateOp(lOpType, aOpName,
        [aStart, aLimit, aDelta]);
    end;

    method OpReadVariable(aResource: not nullable Output; aDataType: TF_DataType;
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'ReadVariableOp';
      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [aResource],
        [
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
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
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
    end;

    method OpSub(x, y: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Sub';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpSum(aInput, aReductionIndices: not nullable Output;
      aKeepDims: Boolean := false; aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'Sum';
      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [aInput, aReductionIndices],
        [
          ('keep_dims', aKeepDims)
        ]);
    end;

    method OpTan(x: not nullable Output; aOpName: not nullable String := '')
      : Output;
    begin
      const lOpType: String = 'Tan';
      (nil, result) := InternalCreateOp(lOpType, aOpName, [x]);
    end;

    method OpVarHandle(aDataType: TF_DataType; aShape: not nullable Shape;
      aContainer: not nullable String := ''; aSharedName: not nullable String := '';
      aOpName: not nullable String := ''): Output;
    begin
      const lOpType: String = 'VarHandleOp';
      (nil, result) := InternalCreateOp(
        lOpType,
        aOpName,
        [],
        [ 
          ('container', aContainer),
          ('shared_name', aSharedName),
          ('dtype', TensorFlowDataType(ord(aDataType))),
          ('shape', aShape)
        ]);
    end;

    method Variable(aInitialValue: not nullable Output; aOpName: not nullable String := '')
      : Tuple of (Operation, Output, Output); //(OpAssignVar, OpReadVar, OpVarHandle)
    begin
      var lOpAssignVar: Operation;
      var lOpReadVar, lOpVarHnd: Output;

      using variableScope := WithScope(MakeName('Variable', aOpName)) do begin
        using lStatus := new Status do begin
          var (success, shp) := GetShape(aInitialValue, lStatus);

          if not success then begin
            raise new OpCreateException withOpType('Variable')
              Message(lStatus.Message);
          end;

          using shp do begin
            lOpVarHnd := OpVarHandle(aInitialValue.Type, shp);
          end;

          using assignScope := WithScope('Assign') do begin
            lOpAssignVar := OpAssignVariable(lOpVarHnd, aInitialValue);
            using readScope := WithScope('Read') do begin
              lOpReadVar := OpReadVariable(lOpVarHnd, aInitialValue.Type);
            end;
          end;
        end;
      end;

      result := (lOpAssignVar, lOpReadVar, lOpVarHnd);
    end;
  end;
end.