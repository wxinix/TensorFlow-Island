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
  TensorFlowDataType = assembly enum
  (
    Float              = TF_DataType.FLOAT,
    Double             = TF_DataType.DOUBLE,
    Int32              = TF_DataType.INT32,
    UInt8              = TF_DataType.UINT8,
    Int16              = TF_DataType.INT16,
    Int8               = TF_DataType.INT8,
    String             = TF_DataType.STRING,
    Complex64          = TF_DataType.COMPLEX64,
    Int64              = TF_DataType.INT64,
    Bool               = TF_DataType.BOOL,
    QInt8              = TF_DataType.QINT8,
    QUInt8             = TF_DataType.QUINT8,
    QInt32             = TF_DataType.QINT32,
    BFloat16           = TF_DataType.BFLOAT16,
    QInt16             = TF_DataType.QINT16,
    QUInt16            = TF_DataType.QUINT16,
    UInt16             = TF_DataType.UINT16,
    Complex128         = TF_DataType.COMPLEX128,
    Half               = TF_DataType.HALF,
    Resource           = TF_DataType.RESOURCE,
    Variant            = TF_DataType.VARIANT,
    UInt32             = TF_DataType.UINT32,
    UInt64             = TF_DataType.UINT64
  );

  TensorFlowCode = assembly enum
  (
    Ok                 = TF_Code.TF_OK,
    Cancelled          = TF_Code.TF_CANCELLED,
    Unknown            = TF_Code.TF_UNKNOWN,
    InvalidArgument    = TF_Code.TF_INVALID_ARGUMENT,
    DeadlineExceed     = TF_Code.TF_DEADLINE_EXCEEDED,
    NotFound           = TF_Code.TF_NOT_FOUND,
    AlreadyExists      = TF_Code.TF_ALREADY_EXISTS,
    PermissionDenied   = TF_Code.TF_PERMISSION_DENIED,
    ResourceExhausted  = TF_Code.TF_RESOURCE_EXHAUSTED,
    FailedPrecondition = TF_Code.TF_FAILED_PRECONDITION,
    Aborted            = TF_Code.TF_ABORTED,
    OutOfRange         = TF_Code.TF_OUT_OF_RANGE,
    Unimplemented      = TF_Code.TF_UNIMPLEMENTED,
    Internal           = TF_Code.TF_INTERNAL,
    Unavailable        = TF_Code.TF_UNAVAILABLE,
    DataLoss           = TF_Code.TF_DATA_LOSS,
    Unauthenticated    = TF_Code.TF_UNAUTHENTICATED
  );

  TensorFlowDataTypeSet = public set of TF_DataType;

const
  TensorFlowNumericalTypes: TensorFlowDataTypeSet = 
  [
    TF_DataType.TF_DOUBLE,
    TF_DataType.TF_FLOAT, 
    TF_DataType.TF_INT16, 
    TF_DataType.TF_INT32, 
    TF_DataType.TF_INT64, 
    TF_DataType.TF_INT8, 
    TF_DataType.TF_FLOAT, 
    TF_DataType.TF_UINT32, 
    TF_DataType.TF_UINT64,
    TF_DataType.TF_UINT8
  ];

type
  Helper = public class
  public
    class method TFDataTypeToString(aDataType: TF_DataType): String;
    begin
      var tfDataType := TensorFlowDataType(ord(aDataType));
      result := tfDataType.ToString;
    end;

    class method TFCodeToString(aCode: TF_Code): String;
    begin
      var tfCode := TensorFlowCode(ord(aCode));
      result := tfCode.ToString;
    end;

    class method ToTFDataType(aType: &Type) RaiseOnInvalid(aFlag: Boolean := True): TF_DataType;
    begin
      case aType.Code of
        TypeCodes.Boolean: result := TF_DataType.TF_BOOL;
        TypeCodes.Byte   : result := TF_DataType.TF_UINT8;
        TypeCodes.UInt16 : result := TF_DataType.TF_UINT16;
        TypeCodes.UInt32 : result := TF_DataType.TF_UINT32;
        TypeCodes.UInt64 : result := TF_DataType.TF_UINT64;
        TypeCodes.SByte  : result := TF_DataType.TF_INT8;
        TypeCodes.Int16  : result := TF_DataType.TF_INT16;
        TypeCodes.Int32  : result := TF_DataType.TF_INT32;
        TypeCodes.Int64  : result := TF_DataType.TF_INT64;
        TypeCodes.Single : result := TF_DataType.TF_FLOAT;
        TypeCodes.Double : result := TF_DataType.TF_DOUBLE;
        TypeCodes.String : result := TF_DataType.TF_STRING;
      else
        if aFlag then begin
          raise new UnSupportedTypeException(aType);
        end else begin
          result := -1;
        end;
      end;
    end;

    class method ReadBytesFromFile(aFile: not nullable String): array of Byte;
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

    class method EncodeString(const aValue: NotNull<String>): String;
    begin
      var src := aValue.ToAnsiChars(false); // Do not take care the case aValue empty.
      var src_len := aValue.Length;
      var dst_len := TF_StringEncodedSize(src_len);
      var dst := new AnsiChar[dst_len];

      using lStatus := new Status do begin
        TF_StringEncode(src, src_len, dst, dst_len, lStatus.Handle);
        if lStatus.OK then begin
          result := String.FromPAnsiChars(dst, dst_len);
        end else begin
          raise new StringEncodeException withString(aValue) Error(lStatus.Message);
        end;
      end;
    end;

    class method DecodeString(const aValue: NotNull<String>): String;
    begin
      var src := aValue.ToAnsiChars;
      var src_len := aValue.Length;
      var dst: ^AnsiChar;
      var dst_len: UInt64;

      using lStatus:= new Status do begin
        TF_StringDecode(src, src_len, @dst, @dst_len, lStatus.Handle);
        if lStatus.OK then begin
          result := String.FromPAnsiChars(dst, dst_len);
        end else begin
          raise new StringDecodeException withError(lStatus.Message);
        end;
      end;    
    end;
  end;
end.