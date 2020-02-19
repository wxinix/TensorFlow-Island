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
          result := lGraph.ReadVariableOp(fResource, fReadHandle.Type);
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

    method GetRandomSeeds(aOperationSeed: nullable Integer): Tuple of (GraphSeed: Integer, LocalSeed: Integer);
    begin
      var graphSeed: Integer := if assigned(Seed) then Seed else 1987;
      var localSeed: Integer := if assigned(aOperationSeed) then aOperationSeed else 1976;
      result := (graphSeed, localSeed);
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
      if (aInput.Type = DataType.Bool) then begin
        aInput := NotNull<Output> (Cast(aInput, DataType.Int8));
      end;

      result := Mean(aInput, ReduceDims(aInput, aAxis), aKeepDims, aOperName);
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
            resource := VarHandleOp(aIniValue.Type, shp);
          end;

          using assignScope := WithScope('Assign') do begin
            assignOp := AssignVariableOp(resource, aIniValue);
            using readScope := WithScope('Read') do begin
              readHandle := ReadVariableOp(resource, aIniValue.Type);
            end;
          end;

          AddInitVariable(assignOp);          
          result := new Variable withResource(resource) ReadHandle(readHandle) AssignOp(assignOp);
          if aTrainable then AddTrainableVariable(result);
        end;
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