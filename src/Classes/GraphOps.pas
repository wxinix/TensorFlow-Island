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
  TensorFlow.Island.Api;

type
  Graph = public partial class
  private
     type Attribute = Tuple of (Name: NotNull<String>, Value: NotNull<Object>);
     type AttributeArray = NotNull<array of Attribute>;
     type InputArray = NotNull<array of NotNull<Output>>;
  private
    method CreateOp(const aOpType, aOpName: NotNull<String>; const aInputs: InputArray = [];
      const aAttrs: AttributeArray = []; const aInputList: InputArray = []): Tuple of (Operation, Output);
    begin
      var lOpDesc := new OperationDescription withGraph(self) OpType(aOpType)
        OpName(MakeName(aOpType, aOpName));

      for each x in aInputs do begin
        lOpDesc.AddInput(x);
      end;

      if aInputList.Length > 0 then begin
        lOpDesc.AddInputs(aInputList);
      end;

      for each a in aAttrs do begin
        lOpDesc.SetAttr(a.Name, a.Value);
      end;

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
    method OpAbort(const aErrMsg: NotNull<String> := ''; aNoErrOnExit: Boolean := true;
      aOpName: NotNull<String> := ''): Operation;
    begin
      const lOpType: String = 'Abort';

      (result, nil) := CreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('error_msg', aErrMsg),
          ('exit_without_error', aNoErrOnExit)
        ]);
    end;

   method OpAbs(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Abs';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpAcos(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
     const lOpType: String = 'Acos';
     (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpAdd(x, y: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Add';
      (nil, result) := CreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpAddN(aInputs: InputArray; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'AddN';
      (nil, result) := CreateOp(lOpType, aOpName, aInputs);
    end;

    method OpArgMax(aInput, aDim: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'ArgMax';
      (nil, result) := CreateOp(lOpType, aOpName, [aInput, aDim]);
    end;

    method OpArgMin(aInput, aDim: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'ArgMin';
      (nil, result) := CreateOp(lOpType, aOpName, [aInput, aDim]);
    end;

    method OpAssignVar(aRsrc, aValue: NotNull<Output>; aOpName: NotNull<String> := ''): Operation;
    begin
      const lOpType: String = 'AssignVariableOp';
      (result, nil) := CreateOp(lOpType, aOpName, [aRsrc, aValue]);
    end;

    method OpAsin(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Asin';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpAtan(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Atan';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpConst(aValue: NotNull<Tensor>; aOpName: NotNull<String> := ''): Output; overload;
    begin
      result := OpConst(aValue, aValue.Data.DataType, aOpName);
    end;

    method OpConst(aValue: NotNull<Tensor>; aDataType: TF_DataType; aOpName: NotNull<String> := '')
      : Output; overload;
    begin
      const lOpType: String = 'Const';

      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('value', aValue),
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpCos(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Cos';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpDiv(x, y: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Div';
      (nil, result) := CreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpFill(aDims, aValue: NotNull<Tensor>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Fill';
      var dims := OpConst(aDims) as NotNull<Output>;
      var val := OpConst(aValue) as NotNull<Output>;

      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [dims, val]);
    end;

    method OpInv(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Inv';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpNegate(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Neg';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpMatMul(a, b: NotNull<Output>; transpose_a: Boolean := false;
      transpose_b: Boolean := false; aOpName: NotNull<String> := '')
      : Output;
    begin
      const lOpType: String = 'MatMul';
      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [a, b],
        [
          ('transpose_a', transpose_a),
          ('transpose_b', transpose_b)
        ]);
    end;

    method OpMul(x, y: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Mul';
      (nil, result) := CreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpPlaceholder(aDataType: TF_DataType; aOpName: NotNull<String> := ''): Output; overload;
    begin
      const lOpType: String = 'Placeholder';
      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpPlaceholder(aDataType: TF_DataType; aShape: NotNull<Shape>;
      aOpName: NotNull<String> := ''): Output; overload;
    begin
      const lOpType: String = 'Placeholder';
      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [],
        [
          ('shape', aShape),
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpRange(aStart, aLimit, aDelta: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Range';
      (nil, result) := CreateOp(lOpType, aOpName, [aStart, aLimit, aDelta]);
    end;

    method OpReadVar(aResource: NotNull<Output>; aDataType: TF_DataType;
      aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'ReadVariableOp';
      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [aResource],
        [
          ('dtype', TensorFlowDataType(ord(aDataType)))
        ]);
    end;

    method OpReduceDims(aInput: NotNull<Output>; aAxis: Output := nil): Output;
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

    method OpReduceSum(aInput, aAxis: NotNull<Output>; aKeepDims: Boolean := false;
      aOpName: NotNull<String> := ''): Output;
    begin
      var reductionIndices := OpReduceDims(aInput, aAxis);
      result := OpSum(aInput, reductionIndices, aKeepDims, aOpName);
    end;

    method OpSave(aFilename: NotNull<Output>; aTensorNames: NotNull<Output>; aData: InputArray;
      aOpName: NotNull<String> := ''): Operation;
    begin
      const lOPType: String = 'Save';
      (result, nil) := CreateOp(lOPType, aOpName, [aFilename, aTensorNames], aData);
    end;

    method OpSin(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Sin';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpSub(x, y: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Sub';
      (nil, result) := CreateOp(lOpType, aOpName, [x, y]);
    end;

    method OpSum(aInput, aReductionIndices: NotNull<Output>; aKeepDims: Boolean := false;
      aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Sum';
      (nil, result) := CreateOp(
        lOpType,
        aOpName,
        [aInput, aReductionIndices],
        [
          ('keep_dims', aKeepDims)
        ]);
    end;

    method OpTan(x: NotNull<Output>; aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'Tan';
      (nil, result) := CreateOp(lOpType, aOpName, [x]);
    end;

    method OpVarHandle(aDataType: TF_DataType; aShape: NotNull<Shape>;
      aContainer: NotNull<String> := ''; aSharedName: NotNull<String> := '';
      aOpName: NotNull<String> := ''): Output;
    begin
      const lOpType: String = 'VarHandleOp';
      (nil, result) := CreateOp(
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

    method Variable(aIniValue: NotNull<Output>; aOpName: NotNull<String> := '')
      : Tuple of (Operation, Output, Output); //(OpAssignVar, OpReadVar, OpVarHandle)
    begin
      var lOpAssignVar: Operation;
      var lOpReadVar, lOpVarHnd: Output;

      using variableScope := WithScope(MakeName('Variable', aOpName)) do begin
        using lStatus := new Status do begin
          var (success, shp) := GetShape(aIniValue, lStatus);

          if not success then begin
            raise new OpCreateException withOpType('Variable') Message(lStatus.Message);
          end;

          using shp do begin
            lOpVarHnd := OpVarHandle(aIniValue.DataType, shp);
          end;

          using assignScope := WithScope('Assign') do begin
            lOpAssignVar := OpAssignVar(lOpVarHnd, aIniValue);
            using readScope := WithScope('Read') do begin
              lOpReadVar := OpReadVar(lOpVarHnd, aIniValue.DataType);
            end;
          end;
        end;
      end;

      result := (lOpAssignVar, lOpReadVar, lOpVarHnd);
    end;
  end;
end.