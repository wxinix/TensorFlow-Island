﻿// MIT License
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

  /// <summary>
  /// The Variable class holds Output nodes and Operation node that are used to initialize,
  /// read and assign a value to a variable.   
  /// </summary>
  /// <remarks>
  /// A variable maintains state in the graph across calls to `run()`. Add a variable to
  /// the graph by calling the MakeVariable method.
  /// </remarks>
  Variable = public class
  private
    fResource: Output;
    fReadHandle: Output;
    fAssignOp: Operation; 
  assembly
    constructor withResource(aResource: NotNull<Output>) ReadHandle(aReadHandle: NotNull<Output>)
      AssignOp(aAssignOp: NotNull<Operation>);
    begin
      fAssignOp := aAssignOp;
      fReadHandle:= aReadHandle;
      fResource := aResource; // VariableHandle
    end;
  public
    /// <summary>
    /// Returns the ReadVariableOp that is used to fetch the value of the variable.
    /// </summary>
    method ReadAfter(aDependencies: NotNull<List<Operation>>): Output;
    begin
      if (aDependencies.Count > 0) then begin
        var lGraph := aDependencies[0].Graph;
        using lGraph.WithDependencies(aDependencies) do begin
          result := lGraph.ReadVariableOp(fResource, fReadHandle.OutputType);
        end;
      end else begin
         result := fReadHandle;
      end;
    end;

    operator Implicit(aVar: NotNull<Variable>): Output;
    begin
      result := aVar.Resource;
    end;
    
    property AssignOp: Operation read fAssignOp;
    property ReadHandle: Output read fReadHandle;
    property Resource: Output read fResource;
  end;

  Graph = public partial class
  public
    method &Const(aValue: NotNull<Tensor>; aOperName: String := nil): Output;
    begin
      exit self.Const (aValue, aValue.Data.Type, aOperName);
    end;

    method ConvertShapeToOutput(aShape: NotNull<Shape>): Output;
    begin
      if aShape.NumDims = 0 then begin
        result := &Const(0);
      end else begin
        result := &Const(NotNull<array of Int64>(aShape.ToArray));
      end;
    end;

    method ClipByNorm(x: NotNull<Output>; aClipNorm: NotNull<Output>; aAxes: Output := nil;
      aOperName: String := nil): Output;
    begin
      using newScope := WithScope(MakeName('ClipByNorm', aOperName)) do begin
        var l2norm_inv := Rsqrt(ReduceSum(Mul(x, x), aAxes, true)); // reciprocal sqrt
        var intermediate := Mul(x, aClipNorm);
        
        result := Identity(
          Mul(
            intermediate, 
            Minimum(
              l2norm_inv, 
              &Div(&Const(1.0), aClipNorm))),
          aOperName
        );
      end;
    end;

    method ClipByAverageNorm(x: NotNull<Output>; aClipNorm: NotNull<Output>; 
      aOperName: String := nil): Output;
    begin
      using newScope := WithScope(MakeName('ClipByAverageNorm', aOperName)) do begin
        var n_element := Cast(Size(x), DataType.Float);
        var l2norm_inv := Rsqrt(ReduceSum(Mul(x, x), Range(Rank(x))));
        
        result := Identity(
          Mul(
            Mul(x, aClipNorm), 
            Minimum(
              Mul(l2norm_inv, n_element), 
              &Div(&Const(1.0), aClipNorm))), 
          aOperName        
        );
      end;
    end;

    method Dropout(x: NotNull<Output>; aKeepProb: NotNull<Output>; aNoiseShape: Shape := nil; 
      aSeed: nullable Integer := nil; aOperName: String := nil): Tuple of (Boolean, Output);
    begin
      using newScope := WithScope(MakeName('dropout', aOperName)) do begin
        var success: Boolean;
        if not assigned(aNoiseShape) then (success, aNoiseShape) := GetTensorShape(x);
        if not success then exit (false, nil);

        var shape_ := ConvertShapeToOutput(NotNull<Shape>(aNoiseShape));
        // uniform [keep_prob, 1.0 + keep_prob]
        var random_tensor := &Add(aKeepProb, RandomUniform(shape_, x.OutputType, aSeed));
        var binary_tensor := Floor(random_tensor);
        result := (true, Mul(&Div(x, aKeepProb), binary_tensor));
        // SetTensorShape(result, GetTensorShape(x)); // No need to call this ?
      end;
    end;

    method Dropout(x: NotNull<Output>; aKeepProb: Double; aNoiseShape: Shape := nil; 
      aSeed: nullable Integer := nil; aOperName: String := nil): Tuple of (Boolean, Output);
    begin
      if not (0 <= aKeepProb <= 1) then begin
        raise new ArgumentOutOfRangeException('keep_prob must be a scalar in the range [0,1].');
      end;

      if aKeepProb.Equals(1) then exit (true, x);
      var keep_prob: Output;
      using newScope := WithScope(MakeName('dropout', aOperName)) do begin
        keep_prob := &Const(aKeepProb);       
      end;
      result := Dropout(x, keep_prob, aNoiseShape, aSeed, aOperName);
    end;

    method GetRandomSeeds(aOpSeed: nullable Integer): Tuple of (GraphSeed: Integer, LocalSeed: Integer);
    begin
      var graphSeed: Integer := if assigned(Seed) then Seed else 1987;
      var localSeed: Integer := if assigned(aOpSeed) then aOperationSeed else 1976;
      result := (graphSeed, localSeed);
    end;

    method GlobalNorm(aTensors: NotNull<array of Output>; aOperName: String := nil): Output;
    begin
      using newScope := WithScope(MakeName('GlobalNorm', aOperName)) do begin
        var half_squared_norms := new Output[aTensors.Length];
        for t in aTensors index i do half_squared_norms[i] := L2Loss(t);
        var half_squared_norm := ReduceSum(Stack(half_squared_norms));
        
        result := Sqrt(
          Mul(half_squared_norm, &Const(2.0)), // * 2 because L2Loss
          if assigned(aOperName) then aOperName else 'global_norm');
      end;
    end;

    method MakeVariable(aIniValue: NotNull<Output>; aTrainable: Boolean := false; 
      aOpName: NotNull<String> := ''): Variable;
    begin
      var assignOp: Operation;
      var readHandle, resource: Output;

      using variableScope := WithScope(MakeName('Variable', aOpName)) do begin
        using lStatus := new Status do begin
          var (success, shp) := GetTensorShape(aIniValue, lStatus);

          if not success then begin
            raise new OpCreateException withOpType('Variable') Message(lStatus.Message);
          end;

          using shp do begin
            resource := VarHandleOp(aIniValue.OutputType, shp);
          end;

          using assignScope := WithScope('Assign') do begin
            assignOp := AssignVariableOp(resource, aIniValue);
            using readScope := WithScope('Read') do begin
              readHandle := ReadVariableOp(resource, aIniValue.OutputType);
            end;
          end;

          AddInitVariable(assignOp);          
          result := new Variable withResource(resource) ReadHandle(readHandle) 
            AssignOp(assignOp);
          if aTrainable then AddTrainableVariable(result);
        end;
      end;
    end;

    method ReduceDims(aInput: NotNull<Output>; aAxis: Output := nil): Output;
    begin
      if assigned(aAxis) then exit aAxis;

      using lStatus := new Status do begin
        var (success, shp) := GetTensorShape(aInput, lStatus);
        if success then begin
          if shp.IsFullySpecified then begin
            var arr := new Int32[shp.NumDims];
            for I: Integer := 0 to shp.NumDims - 1 do arr[I] := I;
            result := &Const(arr, DataType.Int32);
          end else begin
            result := Range(&Const(0), &Const(0), &Const(1)); // start, limit, delta
          end;
        end else begin
          raise new OpCreateException withOpType('ReduceDims') Message(lStatus.Message)
        end;
      end;
    end;

    method ReduceSum(aInput: NotNull<Output>; aAxis: Output := nil; aKeepDims: Boolean := false; 
      aOperName: String := nil): Output;
    begin
      result := Sum(aInput, ReduceDims(aInput, aAxis), aKeepDims, aOperName);
    end;

    method ReduceProd(aInput: NotNull<Output>; aAxis: Output := nil; aKeepDims: Boolean := false; 
      aOperName: String := nil): Output;
    begin
      result := Prod(aInput, ReduceDims(aInput, aAxis), aKeepDims, aOperName);
    end;

    method ReduceMean(aInput: NotNull<Output>; aAxis: Output := nil; aKeepDims: Boolean := false; 
      aOperName: String := nil): Output;
    begin
      if (aInput.OutputType = DataType.Bool) then begin
        aInput := NotNull<Output> (Cast(aInput, DataType.Int8));
      end;

      result := Mean(aInput, ReduceDims(aInput, aAxis), aKeepDims, aOperName);
    end;

    method Stack(aValues: NotNull<array of Output>; aAxis: Integer := 0; 
      aOperName: String := nil): Output;
    begin
      var num_dims := GetTensorNumDims(aValues[0]);
      var expanded_num_dims := num_dims + 1;
      
      if not (-expanded_num_dims <= aAxis < expanded_num_dims) then begin
        raise new ArgumentException(
          $'axis={aAxis} not in range [{-expanded_num_dims},{expanded_num_dims}).');
      end;
      result := Pack(aValues, aAxis, aOperName);
    end;

    method Range(aStart: NotNull<Output>; aLimit: Output := nil; aDelta: Output := nil; 
      aDataType: nullable DataType := nil; aOperName: String := nil): Output;
    begin
      if not assigned(aLimit) then begin
        aLimit := aStart;
        aStart := NotNull<Output>(Cast(&Const(0.0), aStart.OutputType));
      end;

      if not assigned(aDelta) then begin
        aDelta := Cast(&Const(1.0), aStart.OutputType);
      end;

      using newScope := WithScope(MakeName('Range', aOperName)) do begin
        if not assigned(aDataType) then begin
          var dtype_hierarchy: array of DataType :=
            [DataType.Int32, DataType.Int64, DataType.Float, DataType.Double];
          
          if ((not dtype_hierarchy.Contains(aStart.OutputType)) or
              (not dtype_hierarchy.Contains(aLimit.OutputType)) or
              (not dtype_hierarchy.Contains(aDelta.OutputType)) )
          then begin
            raise new ArgumentException('Range() invocation with unexpected type.');
          end;

          var dtypes: array of DataType := 
            [aStart.OutputType, aLimit.OutputType, aDelta.OutputType];
          
          var i_max := dtypes.Select(dtype->dtype_hierarchy.ToList.IndexOf(dtype)).Max;
          var inferred_dtype := dtype_hierarchy[i_max];
          
          aStart := NotNull<Output>(Cast(aStart, inferred_dtype));
          aLimit := Cast(aLimit, inferred_dtype);
          aDelta := Cast(aDelta, inferred_dtype);
        end;

        result := Range(aStart, aLimit, aDelta, aOperName);
      end;
    end;

    /// <summary>
    /// Gets or sets the graph random seed.
    /// </summary>
    /// <remarks>
    /// Operations that rely on a random seed actually derive it from two seeds:
    /// the graph-level and operation-level seeds.This sets the graph-level seed.
    /// </remarks>
    property Seed: nullable Integer read write;
  end;
end.