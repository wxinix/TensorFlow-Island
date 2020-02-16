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
  TensorFlowDataTypeSet = public set of DataType;

const
  TensorFlowNumericalTypes: TensorFlowDataTypeSet =
  [
    DataType.Double,
    DataType.Float,
    DataType.Int8,
    DataType.Int16,
    DataType.Int32,
    DataType.Int64,
    DataType.UInt8,
    DataType.UInt16,
    DataType.UInt32,
    DataType.UInt64
  ];

type
  Helper = assembly static class
  public
    method AsEnum<T>(const aStr: NotNull<String>): Tuple of (Boolean, nullable T);   
    begin
      result := (false, nil);
      if not typeOf(T).IsEnum then exit; 
      for each el in typeOf(T).Constants do begin
        if el.Name.Equals(aStr) then exit (true, T(el.Value));
      end;
    end;

    method DecodeString(const aValue: NotNull<String>): String;
    begin
      var src := aValue.ToAnsiChars;
      var src_len := aValue.Length;
      var dst: ^AnsiChar;
      var dst_len: UInt64;

      using lStatus:= new Status do begin
        TF_StringDecode(src, src_len, @dst, @dst_len, lStatus.Handle);
        if lStatus.Ok then begin
          result := String.FromPAnsiChars(dst, dst_len);
        end else begin
          raise new StringDecodeException withError(lStatus.Message);
        end;
      end;
    end;

    method EncodeString(const aValue: NotNull<String>): String;
    begin
      var src := aValue.ToAnsiChars(false); // Do not take care the case aValue empty.
      var src_len := aValue.Length;
      var dst_len := TF_StringEncodedSize(src_len);
      var dst := new AnsiChar[dst_len];

      using lStatus := new Status do begin
        TF_StringEncode(src, src_len, dst, dst_len, lStatus.Handle);
        if lStatus.Ok then begin
          result := String.FromPAnsiChars(dst, dst_len);
        end else begin
          raise new StringEncodeException withString(aValue) Error(lStatus.Message);
        end;
      end;
    end;
  
    method ReadBytesFromFile(aFile: not nullable String): array of Byte;
    begin
      if not File.Exists(aFile) then begin
        raise new BufferFileNotExistException(aFile);
      end;

      using fs := new FileStream(aFile, FileMode.Open, FileAccess.Read) do begin
        if fs.Length > 0 then begin
          result := new Byte[fs.Length];
          fs.Read(result, fs.Length);
        end else begin
          result := nil;
        end;
      end;
    end;

    method ToArray(aList: not nullable array of Output): array of TF_Output;
    begin
      if aList.Length = 0 then exit nil;
      result := new TF_Output[aList.Length];
      for I: Integer := 0 to aList.Length - 1 do begin
        result[I] := aList[I].AsTFOutput;
      end;
    end;

    method ToTensorFlowDataType(aType: &Type) RaiseOnInvalid(aFlag: Boolean := True): DataType;
    begin
      case aType.Code of
        TypeCodes.Boolean: result := DataType.Bool;
        TypeCodes.Byte   : result := DataType.UInt8;
        TypeCodes.UInt16 : result := DataType.UInt16;
        TypeCodes.UInt32 : result := DataType.UInt32;
        TypeCodes.UInt64 : result := DataType.UInt64;
        TypeCodes.SByte  : result := DataType.Int8;
        TypeCodes.Int16  : result := DataType.Int16;
        TypeCodes.Int32  : result := DataType.Int32;
        TypeCodes.Int64  : result := DataType.Int64;
        TypeCodes.Single : result := DataType.Float;
        TypeCodes.Double : result := DataType.Double;
        TypeCodes.String : result := DataType.String;
      else
        if aFlag then begin
          raise new UnSupportedTypeException(aType);
        end else begin
          result := -1;
        end;
      end;
    end;
  end;
end.