﻿// MIT License
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
  RemObjects.Elements.RTL,
  RemObjects.Elements.System,
  TensorFlow.Island.Api,
  TensorFlow.Island.Aspects;

type
  NotNull<T> = public not nullable T;

  DeviceType = public enum
  (
    CPU,
    GPU,
    TPU  
  );

  DeviceAttrs = public class
  private
    fName: String;
    fType: DeviceType;
    fMemoryLimit: UInt64;
  public
    constructor withName(aName: NotNull<String>) &Type(aType: DeviceType) 
      MemoryLimit(aLimit: UInt64); assembly;
    begin
      fName := aName;
      fType := aType;
      fMemoryLimit := aLimit;
    end;

    property &Type: DeviceType read fType;
    property MemoryLimitBytes: UInt64 read fMemoryLimit;
    property Name: String read fName;
  end;
  
  DataType = public enum
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

  TensorFlowCode = public enum
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

  AttrType = public enum
  (
    String             = TF_AttrType.ATTR_STRING,
    Int                = TF_AttrType.ATTR_INT,
    Float              = TF_AttrType.ATTR_FLOAT,
    Bool               = TF_AttrType.ATTR_BOOL,
    &Type              = TF_AttrType.ATTR_TYPE,
    Shape              = TF_AttrType.ATTR_SHAPE,
    Tensor             = TF_AttrType.ATTR_TENSOR,
    Placeholder        = TF_AttrType.ATTR_PLACEHOLDER,
    Func               = TF_AttrType.ATTR_FUNC
  );

  AttrMetaData = public record
  private
    fAttrMetaData: TF_AttrMetadata;
  public
    constructor(aData: TF_AttrMetadata); assembly;
    begin
      fAttrMetaData := aData;
    end;

    operator Implicit(aData: TF_AttrMetadata): AttrMetaData; assembly;
    begin
      result := new AttrMetaData(aData);
    end;

    property IsList: Byte read fAttrMetaData.is_list;
    property ListSize: Int64 read fAttrMetaData.list_size;
    
    property &Type: AttrType 
      read begin
        result := AttrType(ord(fAttrMetaData.type));
      end;

    property TotalSize: Int64 read fAttrMetaData.total_size;
  end;

  ITensorFlowDisposable = public interface(IDisposable)
    property ID: NativeUInt read;
  end;

  TensorFlowDisposable = public abstract class(ITensorFlowDisposable)
  private
    fDisposed: Boolean := false;

    finalizer;
    begin
      if not fDisposed then begin
        Dispose(false);
      end;
    end;
  protected
    method Dispose(aDisposing: Boolean); virtual;
    begin
      fDisposed := true;
    end;

    method CheckAndRaiseOnDisposed;
    begin
      if fDisposed then begin
        raise new ObjectDisposedException(self);
      end
    end;

    method get_ID: NativeUInt; virtual;
    begin
      result := NativeUInt(@self);
    end;
  public
    method Dispose;
    begin
      if not fDisposed then begin
        Dispose(true);
        BoehmGC.SuppressFinalize(self);
      end;
    end;

    property ID: NativeUInt read get_ID;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowDisposableList<T> = public abstract class(TensorFlowDisposable, IEnumerable<T>)
    where T is ITensorFlowDisposable;
  private
    fDisposed: Boolean := false;
  protected
    fList: List<T>; implements IEnumerable<T>;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        for el in fList do el.Dispose();
      end;

      fList.Clear;
      inherited Dispose(aDisposing);
    end;

    constructor withCapacity(aCapacity: Integer);
    begin
      fList := new List<T>(aCapacity);
    end;

    constructor;
    begin
      fList := new List<T>;
    end;
  public
    method &Add(aItem: T);
    begin
      fList.Add(aItem);
    end;

    method ToArray: array of T;
    begin
      result := fList.ToArray;
    end;

    property Count: Integer
      read begin
        result := fList.Count;
      end;

    property Item[i: Integer]: T
      read begin
        result := fList[i];
      end; default;
  end;

  ITensorFlowObject = public interface(ITensorFlowDisposable)
    property Handle: ^Void read;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowObject<T> = public abstract class(TensorFlowDisposable, ITensorFlowObject)
  public
    type DisposeAction<T> = block(aHandle: ^T);
  private
    fOnDispose: DisposeAction<T>;
    fHandle: ^T := nil;
    fDisposed: Boolean := false;
  protected
    constructor withHandle(aHandle: NotNull<^T>) OnDispose(aAction: DisposeAction<T>);
    begin
      fHandle := aHandle;
      fOnDispose := aAction;
    end;

    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        // Derived class should call its managed object's Dispose().
      end;

      if assigned(fOnDispose) then begin
        fOnDispose(fHandle);
        fHandle := nil;
      end;

      inherited Dispose(aDisposing);
    end;

    method get_ID: NativeUInt; override;
    begin
      result := NativeInt(fHandle);
    end;
  public
    property ID: NativeUInt read get_ID;

    property Handle: ^Void
      read begin
        exit fHandle;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowObjectList<T> = public abstract class(TensorFlowDisposableList<T>)
    where T is ITensorFlowObject;
  public
    property Handles: array of ^Void
      read begin
        if fList.Count = 0 then exit nil;
        result := new ^Void[fList.Count];
        for I: Integer := 0 to fList.Count - 1 do begin
          result[I] := fList[I].Handle;
        end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Buffer = public sealed class(TensorFlowObject<TF_Buffer>)
  private
    fData: ^Void := nil;
    fNumBytes: UInt64 := 0;
    fOnDispose: DisposeAction<TF_Buffer> := aHandle->TF_DeleteBuffer(aHandle);
  public
    class method DeallocateBuffer(aData: ^Void; aSize: UInt64);
    begin
      free(aData);
    end;

    constructor withProtoFile(aFile: NotNull<String>);
    begin
      var bytes := Helper.ReadBytesFromFile(aFile);
      constructor withData(bytes) NumBytes(bytes.Length);
    end;

    constructor withString(const aProtoBuf: NotNull<array of AnsiChar>);
    begin
      var proto_len := aProtoBuf.Length;
      var buffer_hnd := TF_NewBufferFromString(aProtoBuf, proto_len);

      fData := buffer_hnd^.data;
      fNumBytes := buffer_hnd^.length;
      inherited constructor withHandle(buffer_hnd) OnDispose(fOnDispose);
    end;

    constructor withHandle(aHandle: ^TF_Buffer); assembly;
    begin
      fData := aHandle^.data;
      fNumBytes := aHandle^.length;
      inherited constructor withHandle(aHandle) OnDispose(fOnDispose);
    end;

    constructor withData(aData: ^Void) NumBytes(aNumBytes: UInt64);
    begin
      fNumBytes := aNumBytes;
      fData := malloc(fNumBytes);
      memcpy(fData, aData, fNumBytes);
      var buffer_hnd := TF_NewBuffer();
      buffer_hnd^.data := fData;
      buffer_hnd^.length := fNumBytes;
      buffer_hnd^.data_deallocator := @DeallocateBuffer;

      inherited constructor withHandle(buffer_hnd) OnDispose(fOnDispose);
    end;

    method ToArray: array of Byte;
    begin
      if fNumBytes > 0 then begin
        result := new Byte[fNumBytes];
        memcpy(result, fData, fNumBytes);
      end else begin
        result := nil;
      end;
    end;

    property NumBytes: UInt64
      read begin
        result:= fNumBytes;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Operation = public sealed class(TensorFlowObject<TF_Operation>)
  private
    fGraph: Graph;
  public
    constructor withHandle(aHandle: ^TF_Operation) Graph(aGraph: NotNull<Graph>); assembly;
    begin
      fGraph := aGraph;
      inherited constructor withHandle(aHandle) OnDispose(nil);
    end;

    method GetAttrMetaData(const aAttrName: NotNull<String>; aStatus: Status := nil)
      : Tuple of (Boolean, nullable AttrMetaData);
    begin
      using lStatus := new Status do begin
        var meta_data := TF_OperationGetAttrMetadata(Handle, aAttrName.ToAnsiChars(true), lStatus.Handle);

        if lStatus.Ok then begin
          result := (true, meta_data)
        end else begin
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetAttrValue<T>(const aAttrName: NotNull<String>; aStatus: Status := nil)
      : Tuple of (Boolean, nullable T);
    begin
      using lStatus := new Status do begin
        var (success, attr_meta_data) := GetAttrMetaData(aAttrName, lStatus);        
        if not success then begin
          if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
          exit (false, nil);
        end;

        case typeOf(T).GetHashCode of
          typeOf(String).GetHashCode:
          begin
            var (_success, val) := GetAttrString(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of String).GetHashCode:
          begin
            var (_success, val) := GetAttrStringList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(Boolean).GetHashCode:
          begin
            var (_success, val) := GetAttrBool(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of Boolean).GetHashCode:
          begin
            var (_success, val) := GetAttrBoolList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(Int64).GetHashCode:
          begin
            var (_success, val) := GetAttrInt(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of Int64).GetHashCode:
          begin
            var (_success, val) := GetAttrIntList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(Single).GetHashCode:
          begin
            var (_success, val) := GetAttrFloat(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of Single).GetHashCode:
          begin
            var (_success, val) := GetAttrFloatList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(DataType).GetHashCode:
          begin
            var (_success, val) := GetAttrType(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of DataType).GetHashCode:
          begin
            var (_success, val) := GetAttrTypeList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(Shape).GetHashCode:
          begin
            var (_success, val) := GetAttrShape(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of Shape).GetHashCode:
          begin
            var (_success, val) := GetAttrShapeList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(Tensor).GetHashCode:
          begin
            var (_success, val) := GetAttrTensor(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of Tensor).GetHashCode:
          begin
            var (_success, val) := GetAttrTensorList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(Buffer).GetHashCode:
          begin
            var (_success, val) := GetAttrTensorShapeProto(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
          typeOf(array of Buffer).GetHashCode:
          begin
            var (_success, val) := GetAttrTensorShapeProtoList(aAttrName, attr_meta_data, lStatus);
            result := (_success, T(val));
          end;
        else   
          raise new InvalidAttrTypeException($'Invalid operation attr type {typeOf(T).Name}');
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetAttrString(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData; 
      aStatus: Status): Tuple of (Boolean, String);
    begin
      var max_length := aAttrMeta.TotalSize;
      var value := new AnsiChar[max_length];
      
      TF_OperationGetAttrString(Handle, aAttrName.ToAnsiChars(true), value, 
        max_length, aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, String.FromPAnsiChars(value, max_length));
      end else begin
        result := (false, nil);
      end;
    end;

    method GetAttrStringList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of String);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new ^AnsiChar[max_values];
      var lengths := new UInt64[max_values];
      var storage_size := aAttrMeta.TotalSize;
      var storage := new AnsiChar[storage_size];
      
      TF_OperationGetAttrStringList(Handle, aAttrName.ToAnsiChars(true), ^^Void(@values[0]), 
        lengths, max_values, storage, storage_size, aStatus.Handle);
      
      if aStatus.Ok then begin
        var strs := new String[max_values];
        for I: Integer := 0 to max_values - 1 do begin
          strs[I] := String.FromPAnsiChars(values[I], lengths[I]);
        end;
        result := (true, strs);
      end else begin
        result := (false, nil);
      end;    
    end;

    method GetAttrBool(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, nullable Boolean);
    begin
      var value: Byte;
      
      TF_OperationGetAttrBool(Handle, aAttrName.ToAnsiChars(true), @value, 
        aStatus.Handle);      
      
      if aStatus.Ok then begin
        result := (true, value <> 0);
      end else begin
        result := (false, nil);
      end;    
    end;

    method GetAttrBoolList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of Boolean);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new Byte[max_values];
      
      TF_OperationGetAttrBoolList(Handle, aAttrName.ToAnsiChars(true), values, 
        max_values, aStatus.Handle);
      
      if aStatus.Ok then begin
        var bools := new Boolean[max_values];
        for I: Integer := 0 to max_values - 1 do bools[I] := values[I] <> 0;
        result := (true, bools);
      end else begin
        result := (false, nil);
      end;    
    end;

    method GetAttrInt(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, nullable Int64);
    begin
      var value: Int64;
      
      TF_OperationGetAttrInt(Handle, aAttrName.ToAnsiChars(true), @value, 
        aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, value);
      end else begin
        result := (false, nil);
      end;     
    end;

    method GetAttrIntList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of Int64);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new Int64[max_values];
      
      TF_OperationGetAttrIntList(Handle, aAttrName.ToAnsiChars(true), values, 
        max_values, aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, values);
      end else begin
        result := (false, nil);
      end;     
    end;

    method GetAttrFloat(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, nullable Single);
    begin
      var value: Single;
      
      TF_OperationGetAttrFloat(Handle, aAttrName.ToAnsiChars(true), @value, 
        aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, value);
      end else begin
        result := (false, nil);
      end;     
    end;

    method GetAttrFloatList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of Single);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new Single[max_values];
      
      TF_OperationGetAttrFloatList(Handle, aAttrName.ToAnsiChars(true), values, 
        max_values, aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, values);
      end else begin
        result := (false, nil);
      end;    
    end;

    method GetAttrType(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, nullable DataType);
    begin
      var value: TF_DataType;
      TF_OperationGetAttrType(Handle, aAttrName.ToAnsiChars(true), @value, aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, DataType(ord(value)));
      end else begin
        result := (false, nil);
      end;  
    end;

    method GetAttrTypeList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of DataType);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new TF_DataType[max_values];
      
      TF_OperationGetAttrTypeList(Handle, aAttrName.ToAnsiChars(true), values, 
        max_values, aStatus.Handle);
      
      if aStatus.Ok then begin
        var _values := new DataType[max_values];
        for I: Integer := 0 to max_values - 1 do begin
          _values[I] := DataType(ord(values[I]));
        end;
        result := (true, _values);
      end else begin
        result := (false, nil);
      end;    
    end;

    method GetAttrShape(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, nullable Shape);
    begin
      var num_dim := aAttrMeta.TotalSize;
      var value := new Int64[num_dim];

      TF_OperationGetAttrShape(Handle, aAttrName.ToAnsiChars(true), value, 
        num_dim, aStatus.Handle);
     
      if aStatus.Ok then begin
        result := (true, new Shape withDims(value));
      end else begin
        result := (false, nil);
      end;
    end;

    method GetAttrShapeList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of Shape);
    begin
      var num_shapes := aAttrMeta.ListSize;
      var dims := new ^Int64[num_shapes];
      var num_dims := new Int32[num_shapes];
      var storage_size := aAttrMeta.TotalSize;
      var storage := new Int64[storage_size];

      TF_OperationGetAttrShapeList(Handle, aAttrName, @dims[0], num_dims, 
        num_shapes, storage, storage_size, aStatus.Handle);

      if aStatus.Ok then begin
        var shapes := new Shape[num_shapes];
        for I: Integer := 0 to num_shapes - 1 do begin
          if num_dims[I] >= 0 then
            shapes[I] := new Shape(dims[I], num_dims[I])
          else
            shapes[I] := nil; // num_dims[I] = -1, meaning shape is unknown. 
        end;
        result := (true, shapes);
      end else begin
        result := (false, nil);
      end;
    end;

    method GetAttrTensor(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, Tensor);
    begin
      var tensor_hnd: ^TF_Tensor;
      
      TF_OperationGetAttrTensor(Handle, aAttrName.ToAnsiChars(true), @tensor_hnd, 
        aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, new Tensor withHandle(tensor_hnd));
      end else begin
        result := (false, nil);
      end;
    end;

    method GetAttrTensorList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean,  array of Tensor);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new ^TF_Tensor[max_values];

      TF_OperationGetAttrTensorList(Handle, aAttrName.ToAnsiChars(true), values,
        max_values, aStatus.Handle);

      if aStatus.Ok then begin
        var tensors := new Tensor[max_values];
        for I: Integer := 0 to max_values - 1 do begin
          tensors[I] := new Tensor withHandle(values[I]);
        end;
        result := (true, tensors);
      end else begin
        result := (false, nil);
      end;
    end;

    method GetAttrTensorShapeProto(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, nullable Buffer);
    begin
      var value: ^TF_Buffer;

      TF_OperationGetAttrTensorShapeProto(Handle, aAttrName.ToAnsiChars(true), 
        value, aStatus.Handle);

      if aStatus.Ok then begin
        result := (true, new Buffer withHandle(value));
      end else begin
        result := (false, nil);
      end;
    end; 

    method GetAttrTensorShapeProtoList(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData;
      aStatus: Status): Tuple of (Boolean, array of Buffer);
    begin
      var max_values := aAttrMeta.ListSize;
      var values := new ^TF_Buffer[max_values];

      TF_OperationGetAttrTensorShapeProtoList(Handle, aAttrName.ToAnsiChars(true), 
        values, max_values, aStatus.Handle);
      
      if aStatus.Ok then begin
        var buffers := new Buffer[max_values];
        for I: Integer := 0 to max_values - 1 do begin
          buffers[I] := new Buffer withHandle(values[I]);
          result := (true, buffers);
        end;
      end else begin
        result := (false, nil);
      end;
    end; 

    method GetAttrValueProto(const aAttrName: NotNull<String>; aAttrMeta: AttrMetaData; 
      aStatus: Status): Tuple of (Boolean, Buffer);
    begin
      var output_attr_value: ^TF_Buffer;

      TF_OperationGetAttrValueProto(Handle, aAttrName.ToAnsiChars(true), 
         output_attr_value, aStatus.Handle);
      
      if aStatus.Ok then begin
        result := (true, new Buffer withHandle(output_attr_value));
      end else begin
        result := (false, nil);
      end;
    end; 

    method GetOutputListLength(const aArgName: NotNull<String>; aStatus: Status := nil): Integer;
    begin
      using lStatus := new Status do begin
        result := TF_OperationOutputListLength(Handle, aArgName.ToAnsiChars(true), lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetInputListLength(const aArgName: NotNull<String>; aStatus: Status := nil): Integer;
    begin
      using lStatus := new Status do begin
        result := TF_OperationInputListLength(Handle, aArgName.ToAnsiChars(true), lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method ToNodeDef: Tuple of (Boolean, Buffer);
    begin
      using lStatus := new Status do begin
        var buffer_hnd := TF_NewBuffer;
        TF_OperationToNodeDef(Handle, buffer_hnd, lStatus.Handle);
        if lStatus.Ok then begin
          result := (true, new Buffer withHandle(buffer_hnd));
        end else begin
          TF_DeleteBuffer(buffer_hnd);
          result := (false, nil);
        end;
      end;
    end;

    method ToString: String; override;
    begin
      result := $'Operation: {Name}';
    end;

    property ControlOutputs: List<Operation>
      read begin
        var num_control_outputs := NumControlOutputs;
        var op_hnds := new ^TF_Operation[num_control_outputs];
        TF_OperationGetControlOutputs(Handle, op_hnds, num_control_outputs);
        
        for op_hnd in op_hnds do begin
          result.Add(new Operation withHandle(op_hnd) Graph(fGraph));
        end;
      end;

    property &Graph: Graph
      read begin
        result := fGraph;
      end;
    
    property Name: String
      read begin
        result := String.FromPAnsiChars(TF_OperationName(Handle));
      end;

    property NumControlInputs: Integer
      read begin
        result := TF_OperationNumControlInputs(Handle);
      end;

    property NumControlOutputs: Integer
      read begin
        result := TF_OperationNumControlOutputs(Handle);
      end;

    property NumInputs: Integer
      read begin
        result := TF_OperationNumInputs(Handle);
      end;

    property NumOutputs: Integer
      read begin
        result := TF_OperationNumOutputs(Handle);
      end;

    property OpType: String
      read begin
        result := String.FromPAnsiChars(TF_OperationOpType(Handle));
      end;

    property Outputs[aIndex: Integer]: Output
      read begin
        result := new Output withOp(self) &Index(aIndex);
      end; default;
  end;

  OperationList = public sealed class(TensorFlowObjectList<Operation>)
  public
    constructor withCapacity(aCapacity: Integer);
    begin
      inherited constructor withCapacity(aCapacity);
    end;

    constructor;
    begin
      inherited constructor;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  OperationDescription = public sealed class(TensorFlowObject<TF_OperationDescription>)
  private
    fGraph: Graph;
    fOpType: String;
  public
    constructor withGraph(aGraph: NotNull<Graph>) OpType(aType: NotNull<String>)
      OpName(aName: NotNull<String>);
    begin
      fOpType := aType;
      fGraph := aGraph;

      var opdesc_hnd := TF_NewOperation(aGraph.Handle, aType.ToAnsiChars(true),
        aName.ToAnsiChars(true));
      // OnDispose nil, TF_FinishOption will delete OperationDescription.
      inherited constructor withHandle(opdesc_hnd) OnDispose(nil);
    end;

    method ColocateWith(aOp: NotNull<Operation>);
    begin
      TF_ColocateWith(Handle, aOp.Handle);
    end;

    method SetDevice(aDevice: not nullable String);
    begin
      TF_SetDevice(Handle, aDevice.ToAnsiChars(true));
    end;

    method AddInput(aInput: NotNull<Output>);
    begin
      TF_AddInput(Handle, aInput.AsTFOutput);
    end;

    method AddControlInput(aOp: NotNull<Operation>);
    begin
      TF_AddControlInput(Handle, aOp.Handle);
    end;

    method AddInputs(aInputList: NotNull<array of Output>);
    begin
      var tf_outputs := new TF_Output[aInputList.Length];
      for I: Integer := 0 to aInputList.Length - 1 do begin
        tf_outputs[I] := aInputList[I].AsTFOutput;
      end;

      TF_AddInputList(Handle, tf_outputs, tf_outputs.Length);
    end;

    method FinishOperation(aStatus: Status := nil): Tuple of (Boolean, Operation);
    begin
      using lStatus := new Status do begin
        // Desc ptr gets deleted inside TF_FinishOperation.
        var op_hnd := TF_FinishOperation(Handle, lStatus.Handle);

        if lStatus.Ok then begin
          result := (true, new Operation withHandle(op_hnd) Graph(fGraph))
        end else begin
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method SetAttr(const aName: NotNull<String>; aValue: NotNull<Object>);
    begin
      var lHashCode:= aValue.GetType.GetHashCode;
      case lHashCode of
        typeOf(Boolean).GetHashCode:
          SetAttrBool(aName, aValue as Boolean);
        typeOf(array of Boolean).GetHashCode:
          SetAttrBoolList(aName, aValue as array of Boolean);
        typeOf(Single).GetHashCode:
          SetAttrFloat(aName, aValue as Single);
        typeOf(array of Single).GetHashCode:
          SetAttrFloatList(aName, aValue as array of Single);
        typeOf(Int64).GetHashCode:
          SetAttrInt(aName, aValue as Int64);
        typeOf(array of Int64).GetHashCode:
          SetAttrIntList(aName, aValue as array of Int64);
        typeOf(Tensor).GetHashCode:
          SetAttrTensor(aName, aValue as Tensor);
        typeOf(Shape).GetHashCode:
          SetAttrShape(aName, aValue as Shape);
        typeOf(array of Shape).GetHashCode:
          SetAttrShapeList(aName,  aValue as array of Shape);
        typeOf(DataType).GetHashCode:
          SetAttrType(aName, TF_DataType(ord(aValue as DataType)));
        typeOf(String).GetHashCode:
          SetAttrString(aName, aValue as String);
        typeOf(array of String).GetHashCode:
          SetAttrStringList(aName, aValue as array of String);
        typeOf(TensorList).GetHashCode:
          SetAttrTensorList(aName, aValue as TensorList);
        typeOf(array of Byte).GetHashCode:
          SetAttrValueProto(aName, aValue as array of Byte);
      end;
    end;

    method SetAttrBool(const aName: NotNull<String>; aValue: Boolean);
    begin
      var value: Byte := if aValue then 1 else 0;
      TF_SetAttrBool(Handle, aName.ToAnsiChars(true), value);
    end;

    method SetAttrBoolList(const aName: NotNull<String>; aList: NotNull<array of Boolean>);
    begin
      var values := new Byte[aList.Length];
      for I: Integer := 0 to aList.Length - 1 do begin
        values[I] := if aList[I] then 1 else 0;
      end;

      TF_SetAttrBoolList(Handle, aName.ToAnsiChars(true), values, values.Length);
    end;

    method SetAttrFloat(const aName: not nullable String; aValue: Single);
    begin
      TF_SetAttrFloat(Handle, aName.ToAnsiChars(true), aValue);
    end;

    method SetAttrFloatList(const aName: NotNull<String>; aList: NotNull<array of Single>);
    begin
      TF_SetAttrFloatList(Handle, aName.ToAnsiChars(true), aList, aList.Length);
    end;

    method SetAttrFuncName(const aName: NotNull<String>; const aFuncName: NotNull<String>);
    begin
      TF_SetAttrFuncName(Handle, aName.ToAnsiChars(true), aFuncName.ToAnsiChars, aFuncName.Length);
    end;

    method SetAttrInt(const aName: NotNull<String>; aValue: Int64);
    begin
      TF_SetAttrInt(Handle, aName.ToAnsiChars(true), aValue);
    end;

    method SetAttrIntList(const aName: NotNull<String>; aList: NotNull<array of Int64>);
    begin
      TF_SetAttrIntList(Handle, aName.ToAnsiChars(true), aList, aList.Length);
    end;

    method SetAttrString(const aName: NotNull<String>; aValue: NotNull<String>);
    begin
      var length := lstrlenA(aValue.ToAnsiChars(true));
      TF_SetAttrString(Handle, aName.ToAnsiChars(true), aValue.ToAnsiChars, length);
    end;

    method SetAttrStringList(const aName: NotNull<String>; aList: NotNull<array of String>);
    begin
      var num_values := aList.Length;
      var values: array of array of AnsiChar := new array of AnsiChar[num_values];
      var lengths: array of UInt64 := new UInt64[num_values];

      for I: Integer := 0 to num_values - 1 do begin
        // No null terminator, because length is explicitly given below.
        values[I] := aList[I].ToAnsiChars;
        lengths[I] := aList[I].Length;
      end;

      TF_SetAttrStringList(Handle, aName.ToAnsiChars(true), ^^Void(values),
        lengths, num_values);
    end;

    method SetAttrType(const aName: NotNull<String>; aType: TF_DataType);
    begin
      TF_SetAttrType(Handle, aName.ToAnsiChars(true), aType);
    end;

    method SetAttrTypeList(const aName: NotNull<String>; aTypeList: NotNull<array of TF_DataType>);
    begin
      TF_SetAttrTypeList(Handle, aName.ToAnsiChars(true), aTypeList, aTypeList.Length);
    end;

    method SetAttrShape(const aName: NotNull<String>; aShape: NotNull<Shape>);
    begin
      TF_SetAttrShape(Handle, aName.ToAnsiChars(true), aShape.ToArray, aShape.NumDims);
    end;

    method SetAttrShapeList(const aName: NotNull<String>; aList: NotNull<array of Shape>);
    begin
      var num_shapes: Int32 := aList.Length;
      var dims: array of array of Int64 := new array of Int64[num_shapes];
      var num_dims: array of Int32 := new Int32[num_shapes];

      for I: Integer := 0 to num_shapes - 1 do begin
        dims[I] := aList[I].ToArray;
        num_dims[I] := aList[I].NumDims;
      end;

      TF_SetAttrShapeList(Handle, aName.ToAnsiChars(true), ^^Int64(dims), num_dims, num_shapes);
    end;

    method SetAttrTensor(const aName: NotNull<String>; aTensor: NotNull<Tensor>;
      aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrTensor(Handle, aName.ToAnsiChars(true), aTensor.Handle, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method SetAttrTensorList(const aName: NotNull<String>; aTensorList: NotNull<TensorList>; 
      aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrTensorList(Handle, aName.ToAnsiChars(true), aTensorList.Handles,
          aTensorList.Count, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method SetAttrTensorShapeProto(const aName: NotNull<String>; aProto: NotNull<array of Byte>;
      aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrTensorShapeProto(Handle, aName.ToAnsiChars(true), aProto, 
          aProto.Length, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method SetAttrTensorShapeProtoList(const aName: NotNull<String>; aProtos: NotNull<array of array of Byte>; 
      aProtoLens: NotNull<array of UInt64>; aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrTensorShapeProtoList(Handle, aName.ToAnsiChars(true), ^^Void(aProtos),
          aProtoLens, aProtoLens.Length, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;
    
    method SetAttrValueProto(const aName: NotNull<String>; aProto: NotNull<array of Byte>;
      aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetAttrValueProto(Handle, aName.ToAnsiChars(true), aProto, aProto.Length, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;
    
    property OpType: String
      read begin
        result := fOpType;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Status = public sealed class(TensorFlowObject<TF_Status>)    
  public
    [ThreadLocal] 
    class var &Default: Status := new Status;
  public
    constructor;
    begin
      var status_hnd := TF_NewStatus();
      inherited constructor withHandle(status_hnd) OnDispose(aHandle->TF_DeleteStatus(aHandle));
    end;

    method SetCode(aCode: TensorFlowCode) withMessage(const aMsg: String);
    begin
      if not String.IsNullOrEmpty(aMsg) then
        TF_SetStatus(Handle, TF_Code(ord(aCode)), aMsg.ToAnsiChars(true))
      else
        TF_SetStatus(Handle, TF_Code(ord(aCode)), nil);
    end;

    method ToString: String; override;
    begin
      exit $'[Status: Code={Code.ToString}, Message={Message}]';
    end;

    class method ForwardOrCreate(aIncoming: Status): Status;
    begin
      result := if assigned(aIncoming) then aIncoming else new Status;
    end;

    property Ok: Boolean
      read begin
        result := ord(Code) = ord(TF_Code.TF_OK);
      end;

    property Code: TensorFlowCode
      read begin
        result := TensorFlowCode(ord(TF_GetCode(Handle)));
      end;

    property Message: String
      read begin
        result := String.FromPAnsiChars(TF_Message(Handle));
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  RestorableObjectCache = public abstract class(TensorFlowDisposable)
  public
    type RestoreAction = block(const aSavedObject: NotNull<Object>);
  private
    fOnRestore: RestoreAction;
    fCachedObject: Object;
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
      end;

      if assigned(fOnRestore) then begin
        fOnRestore(fCachedObject);
        fOnRestore := nil;
      end;

      inherited Dispose(aDisposing);
    end;

    constructor withCachedObject(const aObject: NotNull<Object>) OnRestore(aAction: RestoreAction);
    begin
      fCachedObject := aObject;
      fOnRestore := aAction;
    end;
  end;

  Scope = public sealed class(RestorableObjectCache)
  public
    constructor withScopeName(const aScopeName: NotNull<String>) OnRestore(aAction: RestoreAction);
    begin
      inherited constructor withCachedObject(aScopeName) OnRestore(aAction);
    end;
  end;
  
  Device = public sealed class(RestorableObjectCache)
  public
    constructor withDeviceName(const aDeviceName: NotNull<String>) OnRestore(aAction: RestoreAction);
    begin
      inherited constructor withCachedObject(aDeviceName) OnRestore(aAction);
    end;
  end;

  Dependencies = public sealed class(RestorableObjectCache)
  public
    constructor withOperations(aOps: NotNull<List<Operation>>) OnRestore(aAction: RestoreAction);
    begin
      inherited constructor withCachedObject(aOps) OnRestore(aAction);
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Shape = public sealed class(TensorFlowDisposable)
  private
    fDims: array of Int64;
    fNumDims: Int32;
    fSize: UInt64;
  public
    constructor withDims(aDims: array of Int64); // aDims can be nil.
    begin
      fNumDims := if assigned(aDims) then aDims.Length else 0;
      fDims := aDims;

      if NumDims> 0 then begin
        fSize := 1;
        for _dim in fDims do fSize := fSize * _dim;
      end else begin
        fSize := 1;
      end;
    end;

    constructor(aDims: ^Int64; aNumDim: Int32); assembly;
    begin
      var dims: array of Int64 := nil;     
      
      if aNumDim > 0 then begin
        dims := new Int64[aNumDim];
        memcpy(dims, aDims, fNumDims * sizeOf(Int64));
      end;

      constructor withDims(dims);
    end;

    method ToArray: array of Int64;
    begin
      result := fDims;
    end;

    method ToString: String; override;
    begin
      var dims_str: String := '';
      for I: Integer := 0 to fNumDims - 1 do begin
        dims_str :=
          if (I < fNumDims - 1) then
            dims_str + $'Dimension({fDims[I]}), '
          else
            dims_str + $'Dimension({fDims[I]})';
      end;
      result := $'TensorShape([{dims_str}])';
    end;

    operator Implicit(aDims: array of Int64): Shape;
    begin
      result := new Shape withDims(aDims);
    end;

    operator Add(aLeft, aRight: NotNull<Shape>): Shape;
    begin
      var new_dims := new Int64[aLeft.NumDims + aRight.NumDims];
      memcpy(@new_dims[0], aLeft.ToArray, aLeft.NumDims);
      memcpy(@new_dims[aLeft.NumDims], aRight.ToArray, aRight.NumDims);
      result := new Shape withDims(new_dims);
    end;

    method AsTensor: Tensor;
    begin
      if fNumDims > 0 then begin
        var arr: NotNull<array of Int64> := new Int64[fNumDims];
        memcpy(arr, fDims, fNumDims);
        result := arr;
      end else begin
        result := nil;
      end;
    end;

    property Dim[aIndex: Int32]: Int64
      read begin
        if (fNumDims > 0) and (0 <= aIndex < fNumDims) then begin
          result := fDims[aIndex]
        end else begin
          raise new InvalidShapeDimIndexException(aIndex, fNumDims);
        end;
      end;

    property NumDims: Int32
      read begin
        result := fNumDims;
      end;

    property Size: UInt64
      read begin
        result := fSize;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Output = public sealed class(TensorFlowDisposable)
  private
    fIndex: Integer;
    fOper: Operation;
  public
    constructor withOp(aOp: NotNull<Operation>) Index(aIndex: Integer = 0);
    begin
      fIndex := aIndex;
      fOper := aOp;
    end;

    method AsTFOutput: TF_Output; assembly;
    begin
      result.oper  := self.Oper.Handle;
      result.index := self.Index;
    end;

    method ToString: String; override;
    begin
      result := $'[Output: Operation = {self.Oper.ToString} Index = {self.Index}]';
    end;

    property &Index: Integer
      read begin
        result := fIndex;
      end;

    property NumConsumers: Integer
      read begin
        result := TF_OperationOutputNumConsumers(AsTFOutput());
      end;

    property Oper: Operation
      read begin
        result := fOper;
      end;

    property &Type: DataType
      read begin
        result := DataType(ord(TF_OperationOutputType(AsTFOutput)));
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  InputList = public sealed class(TensorFlowDisposableList<Output>)
  assembly
    method ToInputArray: array of TF_Output;
    begin
      result := Helper.ToArray(self.ToArray);
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  OutputList = public sealed class(TensorFlowDisposableList<Output>)
  assembly
    method ToOutputArray: array of TF_Output;
    begin
      result := Helper.ToArray(self.ToArray);
    end;
  end;

  // Returns a string because we don't want to directly set TF_WhileParams.name
  // which is a ^AnsiChar, inside the callback. If we do that, we'll have to manage
  // the life time of that pointer. We use/expose TF_WhileParams directly without 
  // any wrap-up because copying back-and-forth between raw TF types and TF-Island
  // types would not be good for performance reason.
  WhileConstructor = public block(var aWhileParam: TF_WhileParams): String;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Graph = public sealed partial class(TensorFlowObject<TF_Graph>)
  private
    fCurrentNameScope: NotNull<String> := '';
    fCurrentDependencies: List<Operation> := new List<Operation>;
    fDeviceName: NotNull<String> := '';
    fNamesCache: Dictionary<String, Integer> := new Dictionary<String, Integer>;
    fPendingInitVars: OperationList := new OperationList;
    fDisposed: Boolean := false;

    method MakeUniqueName(const aName: NotNull<String>): String;
    begin
      var seqid := 0;

      if fNamesCache.ContainsKey(aName) then begin
        seqid := fNamesCache[aName];
        inc(seqid);
        fNamesCache[aName] := seqid;
      end else begin
        fNamesCache.Add(aName, seqid);
      end;

      result := $'{aName}_{seqid}';
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fPendingInitVars.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor;
    begin
      var graph_hnd := TF_NewGraph();
      inherited constructor withHandle(graph_hnd) OnDispose(aHandle->TF_DeleteGraph(aHandle));
    end;

    constructor withUnmangedHandle(aHandle: ^TF_Graph);
    begin
      inherited constructor withHandle(aHandle) OnDispose(nil);
    end;

    /// <summary>
    /// Adds a gradient: the operations needed to compute the partial derivatives of sum of <paramref name="y"/>` wrt to <paramref name="x"/>.
    /// </summary>
    /// <returns>The partial derivatives, the size of the array is the same as the length of the <paramref name="y"/> array.</returns>
    /// <param name="y">The y elements.</param>
    /// <param name="x">The x elements.</param>
    /// <param name="dx">Initial gradients, which represent the symbolic partial derivatives of some loss function `L` w.r.t. <paramref name="y"/> ).   
    /// If the parameter is null, the implementation will use dx for 'OnesLike' for all shapes in <paramref name="y"/></param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// d(y[0] + y[1]+ ...)/dx[0], d(y[0] + y[1] + ...)/dx[1]z...
    /// </remarks>
    method AddGradient(y: NotNull<OutputList>; x: NotNull<OutputList>; dx: NotNull<OutputList>; 
      aStatus: Status := nil): Tuple of (Boolean, OutputList);
    begin
      if y.Count <> dx.Count then begin
        var msg := $'AddGradient y [size={y.Count}] and dx [size={dx.Count}] must have the same size.';
        raise new ArgumentException(msg);
      end;

      using lStatus := new Status do begin
        var dy := new TF_Output[x.Count]; // the partial sum derivative returned.
        
        TF_AddGradients(Handle, y.ToOutputArray, y.Count, x.ToOutputArray, x.Count, 
          dx.ToOutputArray, lStatus.Handle, dy);
          
        if not lStatus.Ok then begin
          var result_list := new OutputList withCapacity(dy.Length);
          for el in dy do begin
            var op := new Operation withHandle(el.oper) Graph(self);
            result_list.Add(new Output withOp(op) &Index(el.index));
          end;
          result := (true, result_list);
        end else begin        
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    /// <summary>
    /// Adds a gradient: the operations needed to compute the partial derivatives of sum of <paramref name="y"/>` wrt to <paramref name="x"/>.
    /// </summary>
    /// <returns>The partial derivatives, the size of the array is the same as the length of the <paramref name="y"/> array.</returns>
    /// <param name="prefix">names the scope into which all gradients operations are being added.  This must be unique within 
    /// the provided graph otherwise this operation will fail. If the value is null, the default prefixing behaviour takes
    /// place, see AddGradients for more details.
    /// </param>
    /// <param name="y">The y elements.</param>
    /// <param name="x">The x elements.</param>
    /// <param name="dx">Initial gradients, which represent the symbolic partial derivatives of some loss function `L` w.r.t. <paramref name="y"/> ).   
    /// If the parameter is null, the implementation will use dx for 'OnesLike' for all shapes in <paramref name="y"/></param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// d(y[0] + y[1]+ ...)/dx[0], d(y[0] + y[1] + ...)/dx[1]z...
    /// </remarks>
    method AddGradientWithPrefix(aPrefix: NotNull<String>; y: NotNull<OutputList>; 
      x: NotNull<OutputList>; dx: NotNull<OutputList>; aStatus: Status := nil)
      : Tuple of (Boolean, OutputList);
    begin
      if y.Count <> dx.Count then begin
        var msg := $'AddGradient y [size={y.Count}] and dx [size={dx.Count}] must have the same size.';
        raise new ArgumentException(msg);
      end;

      using lStatus := new Status do begin
        var dy := new TF_Output[x.Count]; // the partial sum derivative returned.
        
        TF_AddGradientsWithPrefix(Handle, aPrefix.ToAnsiChars(true), y.ToOutputArray,
          y.Count, x.ToOutputArray, x.Count, dx.ToOutputArray, lStatus.Handle, dy);
          
        if not lStatus.Ok then begin
          var result_list := new OutputList withCapacity(dy.Length);
          for el in dy do begin
            var op := new Operation withHandle(el.oper) Graph(self);
            result_list.Add(new Output withOp(op) &Index(el.index));
          end;
          result := (true, result_list);
        end else begin        
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method AddInitVariable(aOp: NotNull<Operation>);
    begin
      for each el in fPendingInitVars do begin
        if el.Equals(aOp) then exit;
      end;

      fPendingInitVars.Add(aOp);
    end;

    method GetFunctions(aStatus: Status := nil): tuple of (Boolean, FunctionList);
    begin
      using lStatus := new Status do begin
        var max_func := NumFunctions;
        var fn_hnds := new ^TF_Function[max_func];
        TF_GraphGetFunctions(Handle, fn_hnds, max_func, lStatus.Handle);
        if lStatus.Ok then begin
          var result_list := new FunctionList withCapacity(max_func);
          for fn_hnd in fn_hnds do begin
            result_list.Add(new TensorFlowFunction withHandle(fn_hnd));
          end;
          result := (true, result_list);
        end else begin
          result := (false, nil);
        end;
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetOperationByName(const aOpName: NotNull<String>): Tuple of (Boolean, Operation);
    begin
      var op_hnd := TF_GraphOperationByName(Handle, aOpName.ToAnsiChars(true));

      if assigned(op_hnd) then begin
        result := (true,
          new Operation withHandle(op_hnd) Graph(self));
      end else begin
        result := (false, nil);
      end;
    end;

    method GetTensorNumDims(aOutput: NotNull<Output>; aStatus: Status := nil): Integer;
    begin
      using lStatus := new Status do begin
        result := TF_GraphGetTensorNumDims(Handle, aOutput.AsTFOutput, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetTensorShape(aOutput: NotNull<Output>; aStatus: Status := nil): Tuple of (Boolean, Shape);
    begin
      using lStatus := new Status do begin
        var nativeOut := aOutput.AsTFOutput;
        var numDims := TF_GraphGetTensorNumDims(Handle, nativeOut, lStatus.Handle);

        if (not lStatus.Ok) then begin
          result := (false, nil);
        end else begin
          if numDims > 0 then begin
            var dims := new Int64[numDims];
            TF_GraphGetTensorShape(Handle, nativeOut, dims, numDims, lStatus.Handle);
            if lStatus.Ok then begin
              result := (true, new Shape withDims(dims));
            end else begin
              result := (false, nil);
            end;
          end else begin
            result := (true, new Shape withDims(nil));
          end;
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method MakeName(const aOpType, aOpName: NotNull<String>): String;
    begin
      var lOpName :=
        if String.IsNullOrEmpty(aOpName) then
          aOpType
        else
          aOpName;

      var name :=
        if String.IsNullOrEmpty(CurrentNameScope) then
          $'{lOpName}'
        else
          $'{CurrentNameScope}/{lOpName}';

      result := MakeUniqueName(name);
    end;

    method ImportGraphDef(aGraphDef: NotNull<Buffer>; aOpts: NotNull<ImportGraphDefOptions>; 
      aStatus: Status := nil); overload;
    begin
      using lStatus := new Status do begin
        TF_GraphImportGraphDef(Handle, ^TF_Buffer(aGraphDef.Handle), aOpts.Handle, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method ImportGraphDef(aBytes: NotNull<array of Byte>; aOpts: NotNull<ImportGraphDefOptions>; 
      aStatus: Status := nil); overload;
    begin
      using graph_def := new Buffer withData(aBytes) NumBytes(aBytes.Length) do begin
        ImportGraphDef(graph_def, aOpts, aStatus);
      end;
    end;

    method ImportGraphDef(aGraphDef: NotNull<Buffer>; aPrefix: NotNull<String> := ''; 
      aStatus: Status := nil); overload;
    begin
      using opts := new ImportGraphDefOptions do begin
        opts.SetPrefix(aPrefix);
        ImportGraphDef(aGraphDef, opts, aStatus);
      end;
    end;

    method ImportGraphDef(aBytes: NotNull<array of Byte>; aPrefix: NotNull<String> := ''; 
      aStatus: Status := nil); overload;
    begin
      using opts := new ImportGraphDefOptions do begin
        using graph_def:= new Buffer withData(aBytes) NumBytes(aBytes.Length) do begin
          opts.SetPrefix(aPrefix);
          ImportGraphDef(graph_def, opts, aStatus);
        end;
      end;
    end;

    method ImportGraphDefWithReturnOutputs(aGraphDef: NotNull<Buffer>; aOpts: NotNull<ImportGraphDefOptions>;
      aStatus: Status := nil): Tuple of (Boolean, OutputList);
    begin
      using lStatus := new Status do begin
        var num_return_outputs := TF_ImportGraphDefOptionsNumReturnOutputs(aOpts.Handle);
        var return_outputs: array of TF_Output := if num_return_outputs > 0 then new TF_Output[num_return_outputs] else nil;
        
        TF_GraphImportGraphDefWithReturnOutputs(Handle, ^TF_Buffer(aGraphDef.Handle), aOpts.Handle,
          return_outputs, num_return_outputs, lStatus.Handle);

        if lStatus.Ok then begin
          var result_list := new OutputList withCapacity(return_outputs.Length);
          for I: Integer := 0 to num_return_outputs - 1 do begin
            var op := new Operation withHandle(return_outputs[I].oper) Graph(self);
            result_list.Add(new Output withOp(op) &Index(return_outputs[I].index));
          end;
          result := (true, result_list);
        end else begin
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method Operations: sequence of Operation; iterator;
    begin
      var pos: UInt64 := 0;
      var op_hnd: ^TF_Operation;
      
      op_hnd := TF_GraphNextOperation(Handle, @pos); 
      while(assigned(op_hnd)) do begin
        yield (new Operation withHandle(op_hnd) Graph(self));
        op_hnd := TF_GraphNextOperation(Handle, @pos); 
      end;
    end;

    method SetTensorShape(aOutput: NotNull<Output>; aShape: NotNull<Shape>; aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_GraphSetTensorShape(Handle, aOutput.AsTFOutput, aShape.ToArray, aShape.NumDims, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method WithDependencies(aNewDependencies: NotNull<List<Operation>>): Dependencies;
    begin
      result := new Dependencies withOperations(fCurrentDependencies) 
        OnRestore(aDependencies->begin 
          fCurrentDependencies := aDependencies as List<Operation> 
        end);

      fCurrentDependencies := fCurrentDependencies.Concat(aNewDependencies).Distinct.ToList;
    end;

    method ToFunction(aName: NotNull<String>; aDesc: NotNull<String>; aOps: NotNull<OperationList>; 
      aInputs: InputList; aOutputs: OutputList; aOutputNames: NotNull<StringList>;
      aAppendHashToName: Boolean := false; aStatus: Status := nil): Tuple of (Boolean, TensorFlowFunction);
    begin
      using lStatus := new Status do begin
        var append_hash_to_fn_name := if aAppendHashToName then 1 else 0;
        var num_opers := aOps.Count;
        var opers := aOps.Handles;
        var ninputs := aInputs.Count;
        var inputs := aInputs.ToInputArray;
        var noutputs := aOutputs.Count;
        var outputs := aOutputs.ToOutputArray;
        var output_names:= aOutputNames.ToAnsiCharPtrs;
        var fn_opts: ^TF_FunctionOptions := nil;

        var fn_hnd := TF_GraphToFunction(
          Handle, 
          aName.ToAnsiChars(true),
          append_hash_to_fn_name,
          num_opers,
          opers,
          ninputs,
          inputs,
          noutputs,
          outputs,
          output_names,
          fn_opts,
          aDesc.ToAnsiChars(true),
          lStatus.Handle);
        
        if lStatus.Ok then begin
          result := (true, new TensorFlowFunction withHandle(fn_hnd));
        end else begin
          result := (false, nil);
        end;
        
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method ToGraphDef(aStatus: Status := nil): Tuple of (Boolean, Buffer);
    begin
      using lStatus := new Status do begin
        var buffer_hnd := TF_NewBuffer;
        TF_GraphToGraphDef(Handle, buffer_hnd, lStatus.Handle);
        
        if lStatus.Ok then begin
          result := (true, new Buffer withHandle(buffer_hnd));
        end else begin
          result := (false, nil);
        end;
        
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method TryEvaluateConstant(aOutput: NotNull<Output>; aStatus: Status := nil): Tuple of (Boolean, Tensor);
    begin
      using lStatus := new Status do begin
        var tensor_hnd: ^TF_Tensor;
        var success := TF_TryEvaluateConstant(Handle, aOutput.AsTFOutput, @tensor_hnd, lStatus.Handle) <> 0;

        if success then begin
          result := (true, new Tensor withHandle(tensor_hnd));
        end else begin
          result := (false, nil);
        end;
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method Versions(aStatus: Status := nil): Tuple of (Boolean, Buffer);
    begin
      using lStatus := new Status do begin
        var buffer_hnd := TF_NewBuffer;
        TF_GraphVersions(Handle, buffer_hnd, lStatus.Handle);
        
        if lStatus.Ok then begin
          result := (true, new Buffer withHandle(buffer_hnd));
        end else begin
          result := (false, nil);
        end;
        
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method &While(aInputs: NotNull<InputList>; aWhileCtor: NotNull<WhileConstructor>; 
      aStatus: Status := nil): Tuple of (Boolean, OutputList);
    begin
      using lStatus := new Status do begin
        var tf_while_params: TF_WhileParams;
        try
          try
            tf_while_params := TF_NewWhile(Handle, aInputs.ToInputArray, aInputs.Count, lStatus.Handle);
            if not lStatus.Ok then exit (false, nil);
            // No need to set tf_while_params.name inside this callback. It will be
            // overwritten anyway.
            var name := aWhileCtor(var tf_while_params); 
            if String.IsNullOrEmpty(name) then name := MakeUniqueName(name);
            tf_while_params.name := name.ToAnsiChars(true); // Overwritten here.
            
            var outputs := new TF_Output[aInputs.Count];
            TF_FinishWhile(@tf_while_params, lStatus.Handle, outputs);
            if not lStatus.Ok then exit (false, nil);

            var result_list := new OutputList withCapacity(outputs.Length);
            for o in outputs do begin
              var op := new Operation withHandle(o.oper) Graph(self);
              result_list.Add(new Output withOp(op) &Index(o.index));
            end;

            result := (true, result_list);
          except
            TF_AbortWhile(@tf_while_params);
          end;
        finally
          if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;
      end;
    end;

    method WithDevice(aNewDeviceName: NotNull<String>): Device;
    begin
      result := new Device withDeviceName(fDeviceName)
        OnRestore(aDeviceName->begin fDeviceName := (aDeviceName as String) end);
      
      if not String.IsNullOrEmpty(fDeviceName) then begin
        raise new DeviceNameAlreadySetException withExistingName(fDeviceName);
      end;

      if not String.IsNullOrEmpty(aNewDeviceName) then begin
        raise new DeviceNameEmptyException;
      end;

      fDeviceName := aNewDeviceName;
    end;

    method WithScope(aNewNameScope: NotNull<String>): Scope;
    begin
      result := new Scope withScopeName(fCurrentNameScope) 
        OnRestore(aScopeName->begin 
          fCurrentNameScope := (aScopeName as String) 
        end);

      if String.IsNullOrEmpty(fCurrentNameScope) then begin
        fCurrentNameScope := aNewNameScope
      end else begin
        fCurrentNameScope := fCurrentNameScope + '/' + aNewNameScope;
      end
    end;

    property CurrentNameScope: String
      read begin
        result := fCurrentNameScope;
      end;

    property CurrentDependencies: List<Operation>
      read begin
        result := fCurrentDependencies;
      end;

    property DeviceName: String
      read begin
        result := fDeviceName;
      end;

    property GlobalVariableInitializer: array of ^TF_Operation
      read begin
        result := fPendingInitVars.Handles;
      end;

    property NumFunctions: Integer
      read begin
        result := TF_GraphNumFunctions(Handle);
      end;

    property &Operation[aName: NotNull<String>]: Operation
      read begin
        (nil, result) := GetOperationByName(aName);
      end; default;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  ImportGraphDefOptions = public class(TensorFlowObject<TF_ImportGraphDefOptions>)
  public
    constructor;
    begin
      var hnd := TF_NewImportGraphDefOptions;
      inherited constructor withHandle(hnd) OnDispose(aHandle->TF_DeleteImportGraphDefOptions(aHandle));
    end;

    method AddControlDependency (aOp: NotNull<Operation>);
    begin
      TF_ImportGraphDefOptionsAddControlDependency(Handle, aOp.Handle);
    end;

    method AddInputMapping (const aSrcName: NotNull<String>; aSrcIndex: Integer; aDst: NotNull<Output>);
    begin
      TF_ImportGraphDefOptionsAddInputMapping(Handle, aSrcName.ToAnsiChars(true), 
        aSrcIndex, aDst.AsTFOutput);
    end;

    method AddReturnOutput (const aOpName: NotNull<String>; aIndex: Integer);
    begin
      TF_ImportGraphDefOptionsAddReturnOutput(Handle, aOpName.ToAnsiChars(true), aIndex);
    end;

    method  RemapControlDependency (const aSrcName: NotNull<String>; aDstOp: NotNull<Operation>);
    begin
      TF_ImportGraphDefOptionsRemapControlDependency(Handle, aSrcName.ToAnsiChars(true), aDstOp.Handle);
    end;

    method SetDefaultDevice(const aDev: NotNull<String>);
    begin
      TF_ImportGraphDefOptionsSetDefaultDevice(Handle, aDev.ToAnsiChars(true));
    end;

    method SetPrefix(const aPrefix: NotNull<String>);
    begin
      TF_ImportGraphDefOptionsSetPrefix(Handle, aPrefix.ToAnsiChars(true));
    end;

    method SetUniquifyNames(aUniquifyNames: Boolean);
    begin
      TF_ImportGraphDefOptionsSetUniquifyNames(Handle, if aUniquifyNames then 1 else 0);
    end;

    method SetUniquifyPrefix(aUniquifyPrefix: Boolean);
    begin
      TF_ImportGraphDefOptionsSetUniquifyPrefix(Handle, if aUniquifyPrefix then 1 else 0);      
    end;

    property NumReturnOutputs: Integer
      read begin
        result := TF_ImportGraphDefOptionsNumReturnOutputs(Handle);
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorData = public class(TensorFlowDisposable)
  private
    fDisposed: Boolean := false;
  protected
    fBytes: ^Void;
    fType: DataType;
    fManaged: Boolean;
    fNumBytes: UInt64;
    fShape: Shape;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fShape.Dispose;
      end;

      if fManaged then begin
        free(fBytes);
      end;

      inherited Dispose(aDisposing);
    end;

    constructor; empty;
  public
    class method DeallocateTensorData(aData: ^Void; aLen: UInt64; aArgs: ^Void);
    begin
      // This does nothing since we use Dispose pattern to manage the data.
      // Also be aware - this user defined deallocator may be IMMEDIATELY called
      // inside TF_NewTensor, as see fit by TensorFlow. This means, later by
      // calling TF_DeleteTensor, this user defined deallocator will not be called.
      // In any case, this class will manage the user-allocated memory if it is
      // constructed using constructor withData. If this class is constructed
      // using constructor withTensorHandle, then the associated data will be managed
      // by whoever creates that raw Tensor pointer.
    end;

    constructor withTensorHandle(aTensorHandle: NotNull<^TF_Tensor>); assembly;
    begin
      fBytes := TF_TensorData(aTensorHandle);
      fType := DataType(ord(TF_TensorType(aTensorHandle)));
      fNumBytes := TF_TensorByteSize(aTensorHandle);

      var num_dims := TF_NumDims(aTensorHandle);
      var dims: array of Int64;
      if num_dims > 0 then begin
        dims := new Int64[num_dims];
        for I: Integer := 0 to num_dims - 1 do begin
          dims[I] := TF_Dim(aTensorHandle, I);
        end;
      end else begin
        dims := nil;
      end;

      fManaged := false;
      fShape := new Shape withDims(dims);
    end;

    constructor (aBytes: ^Void; aDataType: DataType; aNumBytes: Int64;
      aShp: NotNull<Shape>; aManaged: Boolean); private;
    begin
      fBytes := aBytes;
      fType := aDataType;
      fNumBytes := aNumBytes;
      fShape := aShp;
      fManaged := aManaged;
    end;

    /// <summary>
    /// Move the internal data to a new object, while flagging the old object
    /// as disposed. This emulates C++/14 move constructor.
    /// </summary>
    /// <returns></returns>
    method Move: TensorData;
    begin
      fDisposed := true;
      result := new TensorData(fBytes, fType, fNumBytes, fShape, fManaged);
    end;

    property Bytes: ^Void
      read begin
        result := fBytes;
      end;

    property NumBytes: UInt64
      read begin
        result := fNumBytes
      end;

    property &Type: DataType
      read begin
        result := fType
      end;

    property &Shape: Shape
      read begin
        result := fShape
      end;
  end;

  TensorData<T> = public sealed class(TensorData)
  private
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        //
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withValues(aVals: NotNull<array of T>) Dims(aDims: array of Int64);
    begin
      fType := Helper.ToTensorFlowDataType(typeOf(T));
      fShape := new Shape withDims(aDims);
      fManaged := true;

      if aVals.Length <> fShape.Size then begin
        raise new InvalidTensorDataSizeException withDataSize(aVals.Length) DimSize(fShape.Size);
      end;

      if fType <> DataType.String then begin
        fNumBytes := TF_DataTypeSize(TF_DataType(ord(fType))) * aVals.Length;
        fBytes := malloc(fNumBytes);
        if aVals.Length = 1 then begin
          (^T(fBytes))^ := aVals[0];
        end else begin
          memcpy(fBytes, aVals, fNumBytes);
        end;
      end else begin // Special handling for TF_STRING
        fNumBytes := 0;
        for I: Integer := 0 to aVals.Length - 1 do begin
          var str := Helper.EncodeString(String(aVals[I]));
          aVals[I] := T(str); // Replace with encoded string.
          fNumBytes := fNumBytes + str.Length + sizeOf(UInt64); // Offset is UInt64
        end;

        fBytes := malloc(fNumBytes);
        var num_offsets := aVals.Length; // num_offsets equal to aVals.Length
        var offsets := new UInt64[num_offsets];
        var offsets_region_size := num_offsets * sizeOf(UInt64);
        var nbytes := 0;

        for I: Integer := 0 to aVals.Length - 1 do begin
          offsets[I] := offsets_region_size + nbytes;
          memcpy(^Void(^Byte(fBytes) + offsets[I]), String(aVals[I]).ToAnsiChars, String(aVals[I]).Length);
          nbytes := nbytes + String(aVals[I]).Length;
        end;

        if fShape.NumDims = 0 then begin
          memset(fBytes, 0, sizeOf(UInt64));
        end else begin
          memcpy(fBytes, offsets, offsets_region_size);
        end;
      end;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Tensor = public sealed class(TensorFlowObject<TF_Tensor>)
  private
    fData: TensorData;
    fDisposed: Boolean := false;

    class method ConvertToTensor<T>(aVals: NotNull<array of T>): Tensor; overload;
    begin
      var data := new TensorData<T> withValues(aVals) Dims([aVals.Length]);
      result := new Tensor withData(data.Move); // Moved data for Tensor's ownership.
    end;

    class method ConvertToTensor<T>(aVals: NotNull<array of NotNull<array of T>>): Tensor; overload;
    begin
      var height := aVals.Length;
      var width  := aVals[0].Length;

      for I: Integer := 0 to height - 1 do begin
        if aVals[I].Length <> width then begin
          raise new InvalidRectangularTensorData(
            $'Array [Rectangular array height={height} width={width}].' +
            $'Invalid row {I} width={aVals[I].Length}.'
          );
        end;
      end;

      var arr: array of T := new T[height * width];
      for I: Integer := 0 to height - 1 do begin
        memcpy(@arr[I * width], @aVals[I][0], width * sizeOf(T));
      end;

      var data := new TensorData<T> withValues(arr) Dims([height, width]);
      result := new Tensor withData(data.Move); // Moved data for Tensor's ownership.
    end;

    class method ConvertToTensor<T>(aValue: T): Tensor; overload;
    begin
      var data := new TensorData<T> withValues([aValue]) Dims(nil);
      result := new Tensor withData(data.Move); // Moved data for Tensor's ownership.
    end;

  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fData.Dispose; // The internal data gets released here.
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withData(aData: NotNull<TensorData>);
    begin
      var tensor_hnd := TF_NewTensor(
        TF_DataType(ord(aData.Type)),
        aData.Shape.ToArray,
        aData.Shape.NumDims,
        aData.Bytes, // Must be raw bytes; cannot be managed array.
        aData.NumBytes,
        @TensorData.DeallocateTensorData, // does nothing.
        nil);

      if not assigned(tensor_hnd) then raise new TensorCreateException(aData.Type);
      fData := aData;
      inherited constructor withHandle(tensor_hnd) OnDispose(aHandle->TF_DeleteTensor(aHandle));
    end;

    constructor withHandle(aHandle: not nullable ^TF_Tensor); assembly;
    begin
      var lData: TensorData := new TensorData withTensorHandle(aHandle);
      constructor withData(lData);
    end;

    operator Implicit(aValue: Boolean): Tensor;
    begin
      result := ConvertToTensor<Boolean>(aValue);
    end;

    operator Implicit(aValue: Byte): Tensor;
    begin
      result := ConvertToTensor<Byte>(aValue);
    end;

    operator Implicit(aValue: Int16): Tensor;
    begin
      result := ConvertToTensor<Int16>(aValue);
    end;

    operator Implicit(aValue: Integer): Tensor;
    begin
      result := ConvertToTensor<Integer>(aValue);
    end;

    operator Implicit(aValue: Int64): Tensor;
    begin
      result := ConvertToTensor<Int64>(aValue);
    end;

    operator Implicit(aValue: Single): Tensor;
    begin
      result := ConvertToTensor<Single>(aValue);
    end;

    operator Implicit(aValue: Double): Tensor;
    begin
      result := ConvertToTensor<Double>(aValue);
    end;

    operator Implicit(aValue: NotNull<String>): Tensor;
    begin
      result := ConvertToTensor<String>(aValue);
    end;

    operator Implicit(aValues: NotNull<array of Boolean>): Tensor;
    begin
      result := ConvertToTensor<Boolean>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Byte>): Tensor;
    begin
      result := ConvertToTensor<Byte>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Int16>): Tensor;
    begin
      result := ConvertToTensor<Int16>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Integer>): Tensor;
    begin
      result := ConvertToTensor<Integer>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Int64>): Tensor;
    begin
      result := ConvertToTensor<Int64>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Single>): Tensor;
    begin
      result := ConvertToTensor<Single>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of Double>): Tensor;
    begin
      result := ConvertToTensor<Double>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of String>): Tensor;
    begin
      result := ConvertToTensor<String>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Byte>>): Tensor;
    begin
      result := ConvertToTensor<Byte>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Boolean>>): Tensor;
    begin
      result := ConvertToTensor<Boolean>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Int16>>): Tensor;
    begin
      result := ConvertToTensor<Int16>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Integer>>): Tensor;
    begin
      result := ConvertToTensor<Integer>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Int64>>): Tensor;
    begin
      result := ConvertToTensor<Int64>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Single>>): Tensor;
    begin
      result := ConvertToTensor<Single>(aValues);
    end;

    operator Implicit(aValues: NotNull<array of NotNull<array of Double>>): Tensor;
    begin
      result := ConvertToTensor<Double>(aValues);
    end;

    /// <summary>
    /// Convert tensor data to typed scalar value. If the underlying TensorFlow data
    /// type does not match the type parameter, no cast will be performed and nil returned.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    method AsScalar<T>: Tuple of (Boolean, nullable T);
    begin
      var valid_type := (fData.Type = Helper.ToTensorFlowDataType(typeOf(T)) RaiseOnInvalid(false));

      if not (IsScalar and valid_type) then begin
        result := (false, nil);
      end else begin
        if (fData.Type = DataType.String) then begin
          var str: String := Helper.DecodeString(String.FromPAnsiChars(
            ^AnsiChar(^Byte(fData.Bytes) + sizeOf(UInt64)), fData.NumBytes - sizeOf(UInt64)));
          result := (true, T(str));
        end else begin
          var value: T := (^T(fData.Bytes))^;
          result := (true, value);
        end;
      end;
    end;

    /// <summary>
    /// Convert tensor data to typed array. If the underlying TensorFlow data type
    /// does not match the type parameter, no cast will be performed and nil array returned.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    method AsArray<T>: Tuple of (Boolean, array of T);
    begin
      var valid_type := (fData.Type = Helper.ToTensorFlowDataType(typeOf(T)) RaiseOnInvalid(false));

      if not (not IsScalar and valid_type) then begin
        result := (false, nil);
      end else begin
        if (fData.Type = DataType.String) then begin
          var offsets := new UInt64[fData.Shape.Size];
          memcpy(offsets, fData.Bytes, sizeOf(UInt64) * fData.Shape.Size);

          var str_list: List<String> := new List<String>;
          for I: Integer := 0 to offsets.Length - 1 do begin
            var nbytes := if (I <> offsets.Length - 1) then (offsets[I + 1] - offsets[I]) else (fData.NumBytes - offsets[I]);
            var str := String.FromPAnsiChars(^AnsiChar(^Byte(fData.Bytes) + offsets[I]), nbytes);
            str := Helper.DecodeString(str);
            str_list.Add(str);
          end;

          // From String List to array of String. This is to hush compiler.
          var str_arr: array of T := new T[str_list.Count];
          for I: Integer := 0 to str_list.Count - 1 do begin
            str_arr[I] := T(str_list[I]);
          end;
          result := (true, str_arr);
        end else begin
          var values: array of T := new T[fData.Shape.Size];
          memcpy(values, fData.Bytes, fData.NumBytes);
          result := (true, values);
        end;
      end;
    end;

    /// <summary>
    /// Convert the tensor data to string array. If the underlying TensorFlow
    /// data type is TF_STRING, no cast will be performed. Otherwise, numerical
    /// type will be converted to String according to the specified aDecimalDigits
    /// parameter.
    /// </summary>
    /// <param name="aDecimalDigits"></param>
    /// <returns></returns>
    method ConvertDataToStrings(aDecimalDigits: Integer := 1): Tuple of (Boolean, array of String); private;
    begin
      // Local method to convert typed data array to string array.
      method _DoConvertDataToStrings<T>(_aData: array of T): array of String;
      begin
        if not assigned(_aData) then begin
          result := nil
        end else begin
          result := new String[_aData.Length];
        end;

        for I: Integer := 0 to _aData.Length - 1 do begin
          if (typeOf(T) = typeOf(Single)) or (typeOf(T) = typeOf(Double)) then
            result[I] := Double(_aData[I]).ToString(aDecimalDigits)
          else
            result[I] := _aData[I].ToString;
        end;
      end;

      var str_arr := case fData.Type of
        DataType.Bool   : _DoConvertDataToStrings<Boolean>(AsArray<Boolean>()[1]);
        DataType.UInt8  : _DoConvertDataToStrings<Byte>   (AsArray<Byte>   ()[1]);
        DataType.UInt16 : _DoConvertDataToStrings<UInt16> (AsArray<UInt16> ()[1]);
        DataType.UInt32 : _DoConvertDataToStrings<UInt32> (AsArray<UInt32> ()[1]);
        DataType.UInt64 : _DoConvertDataToStrings<UInt64> (AsArray<UInt64> ()[1]);
        DataType.Int8   : _DoConvertDataToStrings<Int8>   (AsArray<Int8>   ()[1]);
        DataType.Int16  : _DoConvertDataToStrings<Int16>  (AsArray<Int16>  ()[1]);
        DataType.Int32  : _DoConvertDataToStrings<Int32>  (AsArray<Int32>  ()[1]);
        DataType.Int64  : _DoConvertDataToStrings<Int64>  (AsArray<Int64>  ()[1]);
        DataType.Float  : _DoConvertDataToStrings<Single> (AsArray<Single> ()[1]);
        DataType.Double : _DoConvertDataToStrings<Double> (AsArray<Double> ()[1]);
        DataType.String : AsArray<String>()[1];
      else nil; end;

      result := (assigned(str_arr), str_arr);
    end;

    method Print(const aDecimalDigits: Integer = 1; const aMaxWidth: Integer = 8): String;
    begin
      const cMaxBytes = 1000; // Tensor cannot exceed this limit in order to print.
      const cAllowedTypes = TensorFlowNumericalTypes + [DataType.String, DataType.Bool];
      // Validate max bytes and allowed types.
      if fData.NumBytes > cMaxBytes then
        exit 'Tensor has {fData.NumBytes} bytes. Too large (>{cMaxBytes}) to print.';
      if not (fData.Type in cAllowedTypes) then
        exit $'Tensor (dtype={fData.Type.ToString}) cannot print.';

      // Convert the tensor data to str_arr. Numerical type will be cast based on aDecimalDigits.
      var (success, str_arr) := ConvertDataToStrings(aDecimalDigits);
      if not success then exit 'Cannot print tensor.';

      // Put high_dims item into one line [v_1, v_2, ..,v_high_dim], inserting each into a seperate str_list
      var high_dim := fData.Shape.Dim[fData.Shape.NumDims - 1];
      var str_list := new List<String>(fData.Shape.Size/high_dim);
      var str: String := '';
      for I: Integer := 0 to str_arr.Length - 1 do begin
        if str_arr[I].Length >= aMaxWidth then exit $'Data item {str_arr[I]} exceeds max width {aMaxWidth}';
        str := str + str_arr[I].PadStart(aMaxWidth, ' ');
        if ((I + 1) mod high_dim = 0) then begin // At each "high-dim" check point
          str_list.Add($'[{str}]'); // Prefix [ and suffix ], then insert into a seperate str_list
          str := '';
        end;
      end;

      // Processt the new str_list: prefix [ and suffix ] with proper white-space.
      for strIndex: Integer := 0 to str_list.Count - 1 do begin
        for dimIndex: Integer := fData.Shape.NumDims - 2 downto 0 do begin
          if ((strIndex + 1) mod fData.Shape.Dim[dimIndex]) =  0 then
            str_list[strIndex] := $'{str_list[strIndex]} ]'
          else
            str_list[strIndex] := $'{str_list[strIndex]}  ';

          if ((strIndex + 1) mod fData.Shape.Dim[dimIndex]) =  1 then
            str_list[strIndex] := $'[ {str_list[strIndex ]}'
          else
            str_list[strIndex] := $'  {str_list[strIndex]}';
        end;
      end;

      // Concat all strings into one string, using line feed.
      result := str_list.JoinedString(#10);
    end;

    property Data: TensorData
      read begin
        result := fData;
      end;

    property IsScalar: Boolean
      read begin
        result := fData.Shape.NumDims = 0;
      end;
  end;

  TensorList = public class(TensorFlowObjectList<Tensor>)
  public
    constructor withCapacity(aCapacity: Integer);
    begin
      inherited constructor withCapacity(aCapacity);
    end;

    constructor;
    begin
      inherited constructor;
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  Session = public sealed class(TensorFlowObject<TF_Session>)
  private    
    fDisposed: Boolean := false;
    fGraph: Graph:= nil;
    fOwnsGraph: Boolean := false; // Graph owned by this Session?
    fDeleteStatus: Status := new Status; 
    fOnDispose: DisposeAction<TF_Session> := aHandle->TF_DeleteSession(aHandle, fDeleteStatus.Handle);    
    fRunner: SessionRunner := nil; // Delayed creation upon access.
    
    method CreateSession(aSessOpts: SessionOptions := nil): Tuple of (Ok: Boolean, Msg: String, Handle: ^TF_Session);
    begin
      using lStatus := new Status do begin
        var opts := if assigned(aSessOpts) then aSessOpts else new SessionOptions;
        var sess_hnd := TF_NewSession(fGraph.Handle, opts.Handle, lStatus.Handle);
        result := (lStatus.Ok, lStatus.Message, sess_hnd);
        if not assigned(aSessOpts) then opts.Dispose;
      end;
    end;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        if fOwnsGraph then fGraph.Dispose;
        fRunner:Dispose; // Colon operator; may have delayed creation.
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor;
    begin
      fGraph := new Graph;
      fOwnsGraph := true;

      var create_session := CreateSession;
      if not create_session.Ok then begin
        raise new SessionCreateException withMessage(create_session.Msg);
      end;

      inherited constructor withHandle(create_session.Handle) OnDispose(fOnDispose);
    end;

    constructor withHandle(aHandle: ^TF_Session) Graph(aGraph: NotNull<Graph>) 
      OwnsGraph(aOwnsGraph: Boolean := false); assembly;
    begin
      fGraph := aGraph;
      fOwnsGraph := aOwnsGraph;
      inherited constructor withHandle(aHandle) OnDispose(fOnDispose)
    end;

    constructor withGraph(aGraph: NotNull<Graph>) Options(aOpts: SessionOptions := nil)
      OwnsGraph(aOwnsGraph: Boolean := false);
    begin
      fGraph := aGraph;
      fOwnsGraph := aOwnsGraph;

      var create_session := CreateSession(aOpts);
      if not create_session.Ok then begin
        raise new SessionCreateException withMessage(create_session.Msg);
      end;

      inherited constructor withHandle(create_session.Handle) OnDispose(fOnDispose);
    end;

    /// <summary>
    /// Close a session. Contacts any other processes associates with session, if 
    /// applicable. May not be called after Dispose.
    /// </summary>
    /// <param name="aStatus">Optional status.</param>
    method CloseSession(aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_CloseSession(Handle, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetTensorInfo(aOutput: NotNull<Output>; aStatus: Status := nil): String;
    begin
      using lStatus := new Status do begin
        var (success, shp) := fGraph.GetTensorShape(aOutput, lStatus);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
  
        if success then begin
          var name := String.FromPAnsiChars(TF_OperationName(aOutput.Oper.Handle));
          result := $'Tensor ("{name}: {aOutput.Index}", ' +
                    $'shape={shp.ToString}, '+
                    $'dtype={aOutput.Type.ToString} )';
        end else begin
          result := '';
        end;
      end;
    end;

    method ListDevices(aStatus: Status := nil): Tuple of (Boolean, IEnumerable<DeviceAttrs>);
    begin
      using lStatus := new Status do begin
        try
          var devlist_hnd := TF_SessionListDevices(Handle, lStatus.Handle);
          if not lStatus.Ok then exit (false, nil);

          var list_size := TF_DeviceListCount(devlist_hnd);
          var devlist := new List<DeviceAttrs>;
          
          for I: Integer := 0 to list_size - 1 do begin
            var nameptr := TF_DeviceListName(devlist_hnd, I, lStatus.Handle);
            if not lStatus.Ok then exit (false, nil);           
            var name := String.FromPAnsiChars(nameptr);

            var typestr_ptr := TF_DeviceListType(devlist_hnd, I, lStatus.Handle);
            if not lStatus.Ok then exit (false, nil);            
            var (nil, dev_type) := Helper.AsEnum<DeviceType>(String.FromPAnsiChars(typestr_ptr));

            var nbytes := TF_DeviceListMemoryBytes(Handle, I, lStatus.Handle);
            if not lStatus.Ok then exit (false, nil);

            devlist.Add(new DeviceAttrs withName(name) &Type(dev_type) MemoryLimit(nbytes));            
          end;

          result := (true, devlist);
        finally
          if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
        end;        
      end;
    end;

    /// <summary>
    /// Creates a session and graph from a model stored in the SavedModel file format.
    /// </summary>		
    /// <remarks>
    /// This function loads the data that was saved using the SavedModel file format, as described
    /// here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
    /// </remarks>
    class method LoadFromSavedModel (aSessOpts: NotNull<SessionOptions>; aRunOptions: Buffer; 
      aExportDir: NotNull<String>; aTags: NotNull<array of String>; aGraph: NotNull<Graph>; 
      aMetaGraphDef: Buffer; aStatus: Status := nil): Tuple of (Boolean, Session);
    begin
      var tags_len := aTags.Length;
      var tags := new ^AnsiChar[tags_len];
      for I: Integer := 0 to tags_len - 1 do tags[I] := aTags[I].ToAnsiChars(true);

      using lStatus := new Status do begin
        var sess_hnd := TF_LoadSessionFromSavedModel(
          aSessOpts.Handle, 
          ^TF_Buffer(aRunOptions:Handle), 
          aExportDir.ToAnsiChars(true), 
          tags, 
          tags_len, 
          aGraph.Handle, 
          ^TF_Buffer(aMetaGraphDef: Handle), 
          lStatus.Handle);

        if lStatus.Ok then begin
          result := (true, new Session withHandle(sess_hnd) Graph(aGraph) OwnsGraph(true));
        end else begin
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;    
    end;

    method RestoreTensor(const aFileName: NotNull<String>; const aTensor: NotNull<String>;
      aDataType: DataType): Output;
    begin
      // To-do
    end;

    method SaveTensors(const aFileName: NotNull<String>; aTensors: array of Tuple of (String, Output)): array of Tensor;
    begin
      // To-do
    end;
    
    property &Graph: Graph
      read begin
        result := fGraph;
      end;

    property Runner: SessionRunner
      read begin
        if not assigned(fRunner) then begin
          fRunner := new SessionRunner withSession(self);
        end;
        fRunner.Reset;
        result := fRunner;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionOptions = public sealed class(TensorFlowObject<TF_SessionOptions>)
  public
    constructor;
    begin
      var sessopts_hnd := TF_NewSessionOptions();
      inherited constructor withHandle(sessopts_hnd) OnDispose(aHandle->TF_DeleteSessionOptions(aHandle));
    end;

    method SetConfig(aProtoData: NotNull<array of Byte>; aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        TF_SetConfig(Handle, aProtoData, aProtoData.Length, lStatus.Handle);
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method SetTarget(aTarget: not nullable String);
    begin
      TF_SetTarget(Handle, aTarget.ToAnsiChars(true));
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionRunnerContext nested in SessionRunner = private class(TensorFlowDisposable)
  private
    fInputs: InputList := new InputList;
    fInputValues: TensorList := new TensorList;
    fOutputs: OutputList := new OutputList;
    fTargets: OperationList := new OperationList;
    fDisposed: Boolean := false;
    fPRunToken: PartialRunToken := nil;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        fInputs.Dispose;
        fInputValues.Dispose;
        fOutputs.Dispose;
        fTargets.Dispose;
        if assigned(fPRunToken) then fPRunToken.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    property Inputs: InputList
      read begin
        result := fInputs;
      end;

    property Outputs: OutputList
      read begin
        result := fOutputs;
      end;

    property InputValues: TensorList
      read begin
        result := fInputValues;
      end;

    property PRunToken: PartialRunToken
      read begin
        result := fPRunToken;
      end
      write begin
        if assigned(fPRunToken) then fPRunToken.Dispose;
        fPRunToken := value;
      end;

    property Targets: OperationList
      read begin
        result := fTargets;
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  PartialRunToken nested in SessionRunner = private class(TensorFlowObject<AnsiChar>)
  public
    constructor withHandle(aHandle: ^AnsiChar); assembly;
    begin
      inherited constructor withHandle(aHandle) OnDispose(aToken->TF_DeletePRunHandle(aToken));
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  SessionRunner = public sealed class(TensorFlowDisposable)
  private
    fSession: Session := nil; // Not created by SessionRunner.
    fContext: SessionRunnerContext := new SessionRunnerContext;
    fDisposed: Boolean := false;
  protected
    method Dispose(aDisposing: Boolean); override;
    begin
      if fDisposed then begin
        exit;
      end else begin
        fDisposed := true;
      end;

      if aDisposing then begin
        // fSession is NOT created by Runner, so donnot dispose Runner.
        // fSession.Dispose;
        fContext.Dispose;
      end;

      inherited Dispose(aDisposing);
    end;
  public
    constructor withSession(aSession: NotNull<Session>);
    begin
      fSession := aSession;
    end;

    method AddInput(aInput: NotNull<Output>; aValue: Tensor): SessionRunner;
    begin
      fContext.Inputs.Add(aInput);
      fContext.InputValues.Add(aValue);
      result := self;
    end;

    method Fetch(aOutput: NotNull<Output>): SessionRunner;
    begin
      fContext.Outputs.Add(aOutput);
      result := self;
    end;

    method AddTarget(aTarget: NotNull<Operation>): SessionRunner;
    begin
      fContext.Targets.Add(aTarget);
      result := self;
    end;

    method Reset;
    begin
      fContext.Dispose;
      fContext := new SessionRunnerContext;
    end;

    method SetupPartialRun(aStatus: Status := nil);
    begin
      using lStatus := new Status do begin
        var token_hnd: ^AnsiChar;
        var inputs        := fContext.Inputs.ToInputArray;
        var ninputs       := fContext.Inputs.Count;
        var outputs       := fContext.Outputs.ToOutputArray;
        var noutputs      := fContext.Outputs.Count;
        var target_opers  := fContext.Targets.Handles;
        var ntargets      := fContext.Targets.Count;

        TF_SessionPRunSetup(fSession.Handle, inputs, ninputs, outputs, noutputs, 
          target_opers, ntargets, @token_hnd, lStatus.Handle);

        if lStatus.Ok then begin
          fContext.PRunToken := new PartialRunToken withHandle(token_hnd);
        end else begin
          raise new PartialRunSetupException withError(lStatus.Message);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method Run(aOp: NotNull<Output>; aStatus: Status := nil): Tensor;
    begin
      Fetch(aOp);
      result := Run(aStatus):Item[0]; // May return nil.
    end;

    method Run(aStatus: Status := nil) MetaData(aMetaData: Buffer := nil)
      Options(aOpts: Buffer := nil): TensorList;
    begin
      using lStatus := new Status do begin
        var run_options   := ^TF_Buffer(aOpts: Handle);
        var inputs        := fContext.Inputs.ToInputArray;
        var input_values  := fContext.InputValues.Handles;
        var ninputs       := fContext.Inputs.Count;
        var outputs       := fContext.Outputs.ToOutputArray;
        var noutputs      := fContext.Outputs.Count;
        var output_values := new ^TF_Tensor[noutputs];
        var target_opers  := fContext.Targets.Handles;
        var ntargets      := fContext.Targets.Count;
        var run_metadata  := ^TF_Buffer(aMetaData:Handle);

        TF_SessionRun(fSession.Handle, run_options, inputs, input_values,
          ninputs, outputs, output_values, noutputs, target_opers,
          ntargets, run_metadata, lStatus.Handle);

        if lStatus.Ok then begin
          result := new TensorList withCapacity(noutputs);
          for I: Integer := 0 to noutputs - 1 do begin
            result.Add(new Tensor withHandle(output_values[I]));
          end;
        end else begin
          writeLn($'SessionRunner.Run failed. Code {ord(lStatus.Code)}. {lStatus.Message}');
          result := nil;
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method PartialRun(aStatus: Status := nil): TensorList;
    begin
      if not assigned(fContext.PRunToken) then begin
        raise new PartialRunTokenNotSetupException;
      end;

      using lStatus := new Status do begin
        var inputs        := fContext.Inputs.ToInputArray;
        var input_values  := fContext.InputValues.Handles;
        var ninputs       := fContext.Inputs.Count;
        var outputs       := fContext.Outputs.ToOutputArray;
        var noutputs      := fContext.Outputs.Count;
        var output_values := new ^TF_Tensor[noutputs];
        var target_opers  := fContext.Targets.Handles;
        var ntargets      := fContext.Targets.Count;
        var token_hnd     := fContext.PRunToken.Handle;

        TF_SessionPRun(fSession.Handle, ^AnsiChar(token_hnd), inputs, input_values,
          ninputs, outputs, output_values, noutputs, target_opers,
          ntargets, lStatus.Handle);

        if lStatus.Ok then begin
          result := new TensorList withCapacity(noutputs);
          for I: Integer := 0 to noutputs - 1 do begin
            result.Add(new Tensor withHandle(output_values[I]));
          end;
        end else begin
          writeLn($'SessionRunner.PartialRun failed. Code {ord(lStatus.Code)}. {lStatus.Message}');
          result := nil;
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;
  end;

  Environment = public sealed static class
  private
    method CheckOsBitSize; static;
    begin
      var os_bit := sizeOf(NativeUInt);
      if os_bit <> sizeOf(UInt64) then begin
        raise new InvalidOsBitSizeException withDetectedOsBitSize(os_bit * sizeOf(Byte));
      end;
    end;
  public
    constructor;
    begin
      CheckOsBitSize;
    end;

    method GetAllOpList: Buffer;
    begin
      result := new Buffer withHandle(TF_GetAllOpList);
    end;

    method GetAllRegisteredKernels(aStatus: Status := nil): Buffer;
    begin
      using lStatus := new Status do begin
        var buffer_hnd := TF_GetAllRegisteredKernels(lStatus.Handle);
        result := if lStatus.Ok then new Buffer withHandle(buffer_hnd) else nil;
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    method GetRegisteredKernelsForOp(aOpName: NotNull<String>; aStatus: Status := nil): Buffer;
    begin
      using lStatus := new Status do begin
        var buffer_hnd := TF_GetRegisteredKernelsForOp(aOpName.ToAnsiChars(true), lStatus.Handle);
        result := if lStatus.Ok then new Buffer withHandle(buffer_hnd) else nil;
        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    property Version: String
      read begin
        result := String.FromPAnsiChars(TF_Version);
      end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowLibrary = public class(TensorFlowObject<TF_Library>)
  public
    constructor withFileName(aName: NotNull<String>);
    begin
      var lStatus := new Status;
      var handle := TF_LoadLibrary(aName.ToAnsiChars(true), lStatus.Handle);

      if not lStatus.Ok then raise new LibraryLoadException withFileName(aName) Message(lStatus.Message);
      inherited constructor withHandle(handle) OnDispose(aHandle->TF_DeleteLibraryHandle(aHandle));
    end;

    method GetOpList: Buffer;
    begin
      var buffer := TF_GetOpList(Handle);
      // buffer memory is owned by the lib_handle, so we do NOT take ownership.
      result := new Buffer withData(buffer.data) NumBytes(buffer.length); 
    end;
  end;

  [TensorFlow.Island.Aspects.RaiseOnDisposed]
  TensorFlowFunction = public class(TensorFlowObject<TF_Function>)
  public
    constructor withHandle(aHandle: ^TF_Function); assembly;
    begin
      inherited constructor withHandle(aHandle) OnDispose(aHnd->TF_DeleteFunction(aHnd));
    end;

    method ToFunctionDef(aStatus: Status := nil): Tuple of (Boolean, Buffer);
    begin
      using lStatus := new Status do begin
        var buffer_hnd := TF_NewBuffer;
        TF_FunctionToFunctionDef(Handle, buffer_hnd, lStatus.Handle);
        
        if lStatus.Ok then begin
          result := (true, new Buffer withHandle(buffer_hnd));
        end else begin
          result := (false, nil);
          TF_DeleteBuffer(buffer_hnd);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    class method ImportFunctionDef(aProto: NotNull<array of Byte>; aStatus: Status := nil)
      : Tuple of (Boolean, TensorFlowFunction);
    begin
      using lStatus := new Status do begin
        var func_hnd := TF_FunctionImportFunctionDef(aProto, aProto.Length, lStatus.Handle);

        if lStatus.Ok then begin
          result := (true, new TensorFlowFunction withHandle(func_hnd));
        end else begin
          result := (false, nil);
        end;

        if assigned(aStatus) then aStatus.SetCode(lStatus.Code) withMessage(lStatus.Message);
      end;
    end;

    property Name: String
      read begin
        result := String.FromPAnsiChars(TF_FunctionName(Handle));
      end;
  end;

  FunctionList = public sealed class(TensorFlowObjectList<TensorFlowFunction>)
  public
    constructor withCapacity(aCapacity: Integer);
    begin
      inherited constructor withCapacity(aCapacity);
    end;

    constructor;
    begin
      inherited constructor;
    end;
  end;

end.