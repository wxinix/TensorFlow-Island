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
  GradientAndVariablePair = Tuple of (Gradient: NotNull<Output>, Variable: NotNull<Variable>);

  Optimizer = public abstract class
  private
    fGraph: Graph;
    fInitialAccumulatorValue: Single;
    fIterations: Variable;
    fLearningRate: Output;
    fOptimizerName: String;
    fUpdateOps: OperationList := new OperationList; protected; readonly;
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

    method InitMoments(aGradientsAndVariables: NotNull<array of GradientAndVariablePair>): OutputList;
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

      using fGraph.WithScope(fOptimizerName) do begin
        fIterations := fGraph.MakeVariable(fGraph.Const(Int64(0)), false, 'iterations');
        var initial_lr := fGraph.Const(aLearningRate);
        var inc_iter_op := fGraph.AssignAddVariableOp(NotNull<Variable>(fIterations), fGraph.Const(Int64(1)));
        fUpdateOps.Add(inc_iter_op);

        using fGraph.WithDependencies([inc_iter_op]) do begin
          fLearningRate := CreateDecayOps(aDecay, initial_lr);
        end;
      end;
    end;

    method ApplyGradient(aGradientsAndVariables: not nullable array of GradientAndVariablePair): OperationList; abstract;

    method ComputeGradient(aLoss: NotNull<Output>; aVariables: array of NotNull<Variable> := nil; aColocateGradientsWithOps: Boolean := false): array of GradientAndVariablePair; virtual;
    begin
      aVariables := if assigned(aVariables) then aVariables else fGraph.TrainableVariables;
      result := new GradientAndVariablePair[aVariables.Length];

      for I: Integer := 0 to aVariables.Length - 1 do begin
        var g: Output := fGraph.AddGradients([aLoss], [aVariables[I].ReadHandle]).Item2.First;
        var v: Variable := aVariables[I];
        result[I] := (g, v);

        if aColocateGradientsWithOps then begin
          var desc_ := new OperationDescription withGraph(fGraph) OpType(g.Oper.OpType) OpName(g.Oper.Name);
          desc_.ColocateWith(v.Resource.Oper);
        end;
      end;
    end;

    method Minimize(aLoss: NotNull<Output>; aVariables: array of NotNull<Variable> := nil): OperationList; virtual;
    begin
      var gv := ComputeGradient(aLoss, aVariables);
      result := ApplyGradient(gv);
    end;

    property &Graph: Graph read fGraph; protected;
    property OptimizerName: String read fOptimizerName; protected;
    property Iterations: Variable read fIterations;
    property LearningRate: Output read fLearningRate;
  end;

  /// <summary>
  /// Stochastic gradient descent optimizer, including support for momentum, learning 
  /// rate decay, and Nesterov momentum.
  /// </summary>
  StochasticGradientDescent = public sealed class(Optimizer)
  private
    fMomentum: Output; readonly;
    fNesterov: Boolean; readonly;
  public
    constructor withGraph(aGraph: NotNull<Graph>) LearningRate(aLearningRate: Single := 0) Momentum(aMomentum: Single := 0) Decay(aDecay: Single := 0) Nesterov(aNesterov: Boolean := false) OpName(aOpName: String := 'SGDOptimizer');
    begin
      inherited constructor(aGraph,  aOpName, aLearningRate, aDecay, 0);

      using aGraph.WithScope(aOpName) do begin
        fMomentum := aGraph.Const(aMomentum, 'Momentum');
      end;

      fNesterov := aNesterov;
    end;

    method ApplyGradient(aGradientsAndVariables: not nullable array of GradientAndVariablePair): OperationList; override;
    begin
      result := new OperationList withCapacity(aGradientsAndVariables.Length);
      var moments := InitMoments(aGradientsAndVariables);

      for gv in aGradientsAndVariables index i do begin
        var lr := Graph.Cast(LearningRate, gv.Gradient.OutputType);
        var m := Graph.Cast(fMomentum, gv.Gradient.OutputType);
        // v = m * moment - lr * g
        var velocity := Graph.Sub(Graph.Mul(m, moments[i]), Graph.Mul(lr, gv.Gradient));
        // moment = v
        fUpdateOps.Add(Graph.Assign(moments[i], velocity).Oper);

        if fNesterov then begin
          // w = w + m * v - lr * g
          var op := Graph.AssignAddVariableOp(gv.Variable, Graph.Mul(lr, Graph.Sub(Graph.Mul(m, velocity), gv.Gradient)));
          fUpdateOps.Add(op);
        end else begin
          // w = w + lr * v
          fUpdateOps.Add(Graph.AssignAddVariableOp(gv.Variable, Graph.Mul(lr, velocity)));
        end;
      end;
    end;

  end;

end.