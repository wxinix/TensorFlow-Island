﻿// MIT License
// Copyright (c) 2019-2021 Wuping Xin.
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

namespace TensorFlow.Island.ApiSamples.SessionRun;

uses
  TensorFlow.Island.Api,
  TensorFlow.Island.Api.Helpers;

type
  Program = class
    class method Main(args: array of String): Int32;
    begin
      // Load graph from a disk protocol buffer file
      var graph := LoadGraph(Environment.CurrentDirectory + '\graph.pb');
      if not assigned(graph) then begin
        writeLn('Cannot load graph');
        exit 1;
      end;

      var status := TF_NewStatus();
      var inpTensor, outTensor: ^TF_Tensor;

      try
        // Create input op
        var inpOp := new TF_Output(
          oper := TF_GraphOperationByName(graph, String('input_4').ToAnsiChars(true)),
          index := 0);

        if not assigned(inpOp.oper) then begin
          writeLn('Cannot init inputOp');
          exit 2;
        end;

        // Create output op
        var outOp := new TF_Output(
          oper := TF_GraphOperationByName(graph, String('output_node0').ToAnsiChars(true)),
          index := 0);

        if not assigned(outOp.oper) then begin
          writeLn('Cannot init outputOp');
          exit 3;
        end;

        // Specify input tensor dimension
        var inpDims := new List<int64_t>([1, 5, 12]);

        // Specify input tensor values
        var inpVals := new List<Single>([
            -0.4809832, -0.3770838, 0.1743573, 0.7720509, -0.4064746, 0.0116595, 0.0051413, 0.9135732, 0.7197526, -0.0400658, 0.1180671, -0.6829428,
            -0.4810135, -0.3772099, 0.1745346, 0.7719303, -0.4066443, 0.0114614, 0.0051195, 0.9135003, 0.7196983, -0.0400035, 0.1178188, -0.6830465,
            -0.4809143, -0.3773398, 0.1746384, 0.7719052, -0.4067171, 0.0111654, 0.0054433, 0.9134697, 0.7192584, -0.0399981, 0.1177435, -0.6835230,
            -0.4808300, -0.3774327, 0.1748246, 0.7718700, -0.4070232, 0.0109549, 0.0059128, 0.9133330, 0.7188759, -0.0398740, 0.1181437, -0.6838635,
            -0.4807833, -0.3775733, 0.1748378, 0.7718275, -0.4073670, 0.0107582, 0.0062978, 0.9131795, 0.7187147, -0.0394935, 0.1184392, -0.6840039 ]);

        // Create input tensor
        inpTensor := CreateTensor(TF_DataType.TF_FLOAT, inpDims, inpVals);

        // Initialize output tensor to nil
        outTensor := nil;

        // Create a new senssion
        var options := TF_NewSessionOptions();
        var session := TF_NewSession(graph, options, status);
        TF_DeleteSessionOptions(options);

        if TF_GetCode(status) <> TF_Code.TF_OK then begin
          writeLn('Cannot create session.');
          exit 4;
        end;

        writeLn('Check Point.');

        // Run session with the supplied input/output ops
        TF_SessionRun(
          session,
          nil, // run options.
          @inpOp, @inpTensor, 1,
          @outOp, @outTensor, 1,
          nil, // target options.
          0,   // num targets.
          nil, // run meta data.
          status);

        if TF_GetCode(status) <> TF_Code.TF_OK then begin
          writeLn('Error run session.');
          exit 6;
        end;

        // Delete session
        TF_DeleteSession(session, status);
        if TF_GetCode(status) <> TF_Code.TF_OK then begin
          writeLn('Error delete session.');
          exit 7;
        end;

        // Retrieve output tensor data
        var data := ^Single(TF_TensorData(outTensor));
        writeLn($'Output vals: {data[0]}, {data[1]}, {data[2]}, {data[3]}');
        readLn;
      finally
        TF_DeleteStatus(status);
        DeleteGraph(graph);
        DeleteTensor(inpTensor);
        DeleteTensor(outTensor);
      end;
    end;
  end;
end.