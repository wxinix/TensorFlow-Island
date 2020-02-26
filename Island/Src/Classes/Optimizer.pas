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
  TensorFlow.Island.Api;

type
  Optimizer = public abstract class
  private
    fGraph: Graph;
    fInitialAccumulatorValue: Single;
    fIterations: Variable;
    fLearningRate: Output;
    fOptimizerName: String;
    fUpdateOps: OperationList := new OperationList; readonly;
  protected
    method CreateDecayOps(aDecay: Single; aInitialLearningRate: NotNull<Output>): Output;
    begin
      if (aDecay <= 0) then begin
        result := aInitialLearningRate;
      end else begin
        var decay := fGraph.Const(aDecay, 'Decay');
        var one := fGraph.Const(Single(1));
        result := fGraph.Mul(
          aInitialLearningRate,
          fGraph.Div(one,
                     fGraph.Add(one,
                                fGraph.Mul(decay,
                                           fGraph.Cast(fGraph.Sub(fIterations.ReadAfter(fGraph.CurrentDependencies.ToArray),
                                                                  fGraph.Const(Int64(1))),
                                                       decay.OutputType)
                                          )
                              )
                    ),
          'learningrate');
      end;
    end;

    method InitMoments(aGradientsAndVariables: not nullable array of Tuple of (Gradient: Output, Variable: Variable)): OutputList;
    begin
      result := new OutputList withCapacity(aGradientsAndVariables.Length);
      for gv in aGradientsAndVariables do begin
        var varType := gv.Variable.ReadHandle.OutputType;
        var (nil, varShape) := fGraph.GetTensorShape(gv.Variable.ReadHandle);
        var v := fGraph.VariableV2(varShape, varType);
        result.Add(v);
        fGraph.AddInitVariable(fGraph.Assign(v, fGraph.Constant(fInitialAccumulatorValue, varShape, varType)).Oper);
      end;
    end;
  public
    constructor(aGraph: NotNull<Graph>; const aOpName: NotNull<String>; aLearningRate, aDecay, aInitialAccumulatorValue: Single);
    begin
      if (aInitialAccumulatorValue < 0) then begin
        raise new ArgumentException($'InitialAccumulatorValue = {aInitialAccumulatorValue}. It must be non-negative.');
      end;

      fGraph := aGraph;
      fInitialAccumulatorValue := aInitialAccumulatorValue;
      fOptimizerName := aOpName;

      using newScope := fGraph.WithScope(fOptimizerName) do begin
        fIterations := fGraph.MakeVariable(fGraph.Const(Int64(0)), false, 'iterations');
        var initialLearningRate := fGraph.Const(aLearningRate);
        var incOp := fGraph.AssignAddVariableOp(NotNull<Variable>(fIterations), fGraph.Const(Int64(1)));
        fUpdateOps.Add(incOp);

        using fGraph.WithDependencies([incOp]) do begin
          fLearningRate := CreateDecayOps(aDecay, initialLearningRate);
        end;
      end;
    end;

    method ApplyGradient(aGradientsAndVariables: not nullable array of Tuple of (Gradient: Output, Variable: Variable)): OperationList; abstract;

    method ComputeGradient(aLoss: NotNull<Output>; aVariables: array of Variable := nil; aColocateGradientsWithOps: Boolean := false): array of Tuple of (Gradient: Output, Variable: Variable); virtual;
    begin
      aVariables := if assigned(aVariables) then aVariables else fGraph.TrainableVariables;
      result := new (Tuple of (Gradient: Output, Variable: Variable))[aVariables.Length];

      for I: Integer := 0 to aVariables.Length - 1 do begin
        var grad := fGraph.AddGradients([aLoss], [aVariables[I].ReadHandle]).Item2.First;
        var var_ := aVariables[I];
        result[I] := (grad, var_);

        if aColocateGradientsWithOps then begin
          var desc_ := new OperationDescription withGraph(fGraph) OpType(grad.Oper.OpType) OpName(grad.Oper.Name);
          desc_.ColocateWith(var_.Resource.Oper);
        end;
      end;
    end;

    method Minimize(aLoss: NotNull<Output>; aVariables: array of Variable := nil): OperationList; virtual;
    begin
      var gv := ComputeGradient(aLoss, aVariables);
      result := ApplyGradient(gv);
    end;

    property &Graph: Graph read fGraph; protected;
    property OptimizerName: String read fOptimizerName; protected;
    property Iterations: Variable read fIterations;
    property LearningRate: Output read fLearningRate;
  end;

end.