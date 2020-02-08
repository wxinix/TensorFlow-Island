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


namespace TensorFlow.Island.ApiSamples.AllocateTensor;

uses
  TensorFlow.Island.Api;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      const dims: array of int64_t = [1, 5, 12];
      var dataSize := sizeOf(Single);      
      
      for each i in dims do 
        dataSize := dataSize * i;

      const data: array of Single = 
        [ 
          -0.4809832, -0.3770838, 0.1743573, 0.7720509, -0.4064746, 0.0116595, 0.0051413, 0.9135732, 0.7197526, -0.0400658, 0.1180671, -0.6829428,
          -0.4810135, -0.3772099, 0.1745346, 0.7719303, -0.4066443, 0.0114614, 0.0051195, 0.9135003, 0.7196983, -0.0400035, 0.1178188, -0.6830465,
          -0.4809143, -0.3773398, 0.1746384, 0.7719052, -0.4067171, 0.0111654, 0.0054433, 0.9134697, 0.7192584, -0.0399981, 0.1177435, -0.6835230,
          -0.4808300, -0.3774327, 0.1748246, 0.7718700, -0.4070232, 0.0109549, 0.0059128, 0.9133330, 0.7188759, -0.0398740, 0.1181437, -0.6838635,
          -0.4807833, -0.3775733, 0.1748378, 0.7718275, -0.4073670, 0.0107582, 0.0062978, 0.9131795, 0.7187147, -0.0394935, 0.1184392, -0.6840039 
        ];

      var tensor := TF_AllocateTensor(TF_DataType.TF_FLOAT, dims, dims.Length, dataSize);
      
      try
        if assigned(tensor) and assigned(TF_TensorData(tensor)) then begin
          memcpy(TF_TensorData(tensor), data, Math.Min(dataSize, TF_TensorByteSize(tensor)))
        end else begin
          writeLn('Wrong create tensor!');
          exit 1;
        end;

        if TF_TensorType(tensor) <> TF_DataType.TF_FLOAT then begin
          writeLn('Wrong tensor type');
          exit 2;
        end;

        if TF_NumDims(tensor) <> dims.Length then begin
          writeLn('Wrong number of dimensions');
          exit 3;
        end;

        for i: Integer := 0 to dims.Length - 1 do begin
          if TF_Dim(tensor, i) <> dims[i] then begin
            writeLn('Wrong dimension size for dim');
            exit 4;
          end;
        end;

        if TF_TensorByteSize(tensor) <> dataSize then begin
          writeLn('Wrong tensor byte size');
          exit 5;
        end;

        var tensor_data := ^Single(TF_TensorData(tensor));
        if not assigned(tensor_data) then begin
          writeLn('Wrong data tensor');
          exit 6;
        end;

        for i: Integer := 0 to data.Length - 1 do begin
          if tensor_data[i] <> data[i] then begin
            writeLn('Element: ' + i.ToString + 'does not match');
            exit 7;
          end;
        end;

        writeLn('Congradulations. Success allocating tensor!');
        writeLn($'TF_BOOL size is {TF_DataTypeSize(TF_DataType.TF_BOOL)}');
        writeLn($'TF_STRING size is {TF_DataTypeSize(TF_DataType.TF_STRING)}');
        readLn;
      finally
        TF_DeleteTensor(tensor);
      end;
    end;
  end;

end.