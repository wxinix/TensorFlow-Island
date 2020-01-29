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

namespace TensorFlow.Island.Samples.InterfaceTest;

uses
  TensorFlow,
  TensorFlow.Island.ApiUtils;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var graph := LoadGraph('C:\DEVLIBS\TensorFlow-Island\exe\Windows\x86_64\graph.pb');
      var status := TF_NewStatus();
      var inputOps, outputOps: List<TF_Output>;
      var inputTensors: List<^TF_Tensor>;
      var outputTensors: List<^TF_Tensor>;
      var session: ^TF_Session;

      try
        if not assigned(graph) then begin
          writeLn('Cannot load graph');
          exit 1;
        end;

        var inputDims := new List<int64_t>([1, 5, 12]);
        var inputVals := new List<Single>(
          [
            -0.4809832, -0.3770838, 0.1743573, 0.7720509, -0.4064746, 0.0116595, 0.0051413, 0.9135732, 0.7197526, -0.0400658, 0.1180671, -0.6829428,
            -0.4810135, -0.3772099, 0.1745346, 0.7719303, -0.4066443, 0.0114614, 0.0051195, 0.9135003, 0.7196983, -0.0400035, 0.1178188, -0.6830465,
            -0.4809143, -0.3773398, 0.1746384, 0.7719052, -0.4067171, 0.0111654, 0.0054433, 0.9134697, 0.7192584, -0.0399981, 0.1177435, -0.6835230,
            -0.4808300, -0.3774327, 0.1748246, 0.7718700, -0.4070232, 0.0109549, 0.0059128, 0.9133330, 0.7188759, -0.0398740, 0.1181437, -0.6838635,
            -0.4807833, -0.3775733, 0.1748378, 0.7718275, -0.4073670, 0.0107582, 0.0062978, 0.9131795, 0.7187147, -0.0394935, 0.1184392, -0.6840039
          ]);
        
        inputOps := new List<TF_Output>(
          [
            new TF_Output(oper := TF_GraphOperationByName(graph, String('input_4').ToAnsiChars(true)), index := 0)
          ]);

        inputTensors := new List<^TF_Tensor>(
          [
            CreateTensor(TF_DataType.TF_FLOAT, inputDims, inputVals)
          ]);

        outputOps := new List<TF_Output>(
          [
            new TF_Output(oper := TF_GraphOperationByName(graph, String('output_node0').ToAnsiChars(true)), index := 0)
          ]);

        outputTensors := new List<^TF_Tensor>(
          [
            nil
          ]);

        session := CreateSession(graph);
        if not assigned(session) then begin
          writeLn('InterfaceTest: Cannot create session.');
          exit 2;
        end;

        result := RunSession(session, inputOps, inputTensors, outputOps, outputTensors);
        if result = TF_Code.TF_OK then begin
          var tensorData := GetTensorData<Single>(outputTensors[0]);
          writeLn($'Output vals: {tensorData[0]}, {tensorData[1]}, {tensorData[2]}');  
        end else begin
          writeLn($'Error run session TF_CODE: {result}');
        end;

        readLn; 
      finally
        DeleteGraph(graph);
        TF_DeleteStatus(status);
        DeleteTensors(inputTensors);
        DeleteTensors(outputTensors);
        DeleteSession(session);
      end;
    end;      
  end;

end.