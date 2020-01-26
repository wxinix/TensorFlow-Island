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


namespace TensorFlow.Island.Samples.TensorInfo;

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

      try
        if not assigned(graph) then begin
          writeLn('Cannot load graph');
          exit 1;
        end;

        PrintTensorInfo(graph, 'input_4', status);
        writeLn('');
        PrintTensorInfo(graph, 'output_node0', status);
        readLn;
      finally
        TF_DeleteStatus(status);
        DeleteGraph(graph);
      end;
    end;
  end;

end.