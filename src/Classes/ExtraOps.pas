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
  public
    method &Const(aValue: NotNull<Tensor>; aOperName: String := nil): Output;
    begin
      exit self.Const (aValue, aValue.Data.Type, aOperName);
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
            result := Range(&Const(0), &Const(0), &Const(1));
          end;
        end else begin
          raise new OpCreateException withOpType('ReduceDims') Message(lStatus.Message)
        end;
      end;
    end;
  end;
end.