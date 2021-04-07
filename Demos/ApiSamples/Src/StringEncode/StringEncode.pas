// MIT License
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


namespace TensorFlow.Island.ApiSamples.StringEncode;

uses
  TensorFlow.Island.Api,
  TensorFlow.Island.Api.Helpers;

type 
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      const str = '123456789012';
      var status := TF_NewStatus;
      var tensor := ScalarStringTensor(str.ToAnsiChars(true), status);
      var byte_size := TF_TensorByteSize(tensor);
      writeLn('Tensor byte size:' + byte_size);
      var tensor_type := TF_TensorType(tensor);
      writeLn('Tensor type:' + DataTypeToString(tensor_type));
      TF_DeleteStatus(status);
      TF_DeleteTensor(tensor);
      readLn();
    end;
  end;

end.