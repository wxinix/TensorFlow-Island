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
  DeviceNameException = public class(Exception)
  public
    constructor withDuplicateName(aName: String);
    begin
      inherited constructor($'Trying to set device name with duplicate name {aName}.');
    end;

    constructor withEmptyName;
    begin
      inherited constructor('Trying to set device name with empty name.');
    end;
  end;

  TensorDataSizeException = public class(Exception)
  public
    constructor withTensorDataSize(aDataSize: Integer) ShapeSize(aShapeSize: Integer);
    begin
      inherited constructor(
        $'Tensor data[size={aDataSize}] inconsistent with shape[size={aShapeSize}].');
    end;
  end;

  OsBitSizeException = public class(Exception)
  public
    constructor withDetectedOsBitSize(aSize: Integer);
    begin
      inherited constructor($'Supports 64bit system only. {aSize}bit system detected.');
    end;
  end;

  LibraryLoadException = public class(Exception)
  public
    constructor withLibName(aName: String) Error(aErr: String);
    begin
      inherited constructor($'Error loading TensorFlow library "{aName}": {aErr}');
    end; 
  end;

  ObjectDisposedException = public class(Exception)
  public
    constructor (aObject: TensorFlowDisposable);
    begin
      var name := aObject.GetType.Name;
      var id := aObject.ID;
      inherited constructor($'{name} instance {id} already disposed.');
    end;
  end;

  OpCreateException = public class(Exception)
  public
    constructor withOpType(aOpType: String) Error(aErr: String);
    begin
      inherited constructor($'Error creating Op[type={aOpType}]: {aErr}.');
    end;
  end;

  PartialRunSetupException = public class(Exception)
  public
    constructor withError(aErr: String);
    begin
      inherited constructor($'Error setting up partial run: {aErr}.');
    end;
  end;

  PartialRunTokenException = public class(Exception)
  public
    constructor;
    begin
      inherited constructor('Partial run has uninitialized token.');
    end;
  end;

  SessionCreateException = public class(Exception)
  public
    constructor withError(aErr: String);
    begin
      inherited constructor(aErr);
    end;
  end;

  StringDecodeException = public class(Exception)
  public
    constructor withError(aErr: String);
    begin
      inherited constructor(aErr);
    end;
  end;

  StringEncodeException = public class(Exception)
  public
    constructor withString(aValue: String) Error(aErr: String);
    begin
      inherited constructor($'Error encoding "{aValue}": {aErr}.');
    end;
  end;

  TensorCreateException = class(Exception)
  public
    constructor withTensorType(aType: DataType);
    begin
      var msg := $'Fail creating tensor[dtype={aType.ToString}].';
      inherited constructor(msg);
    end;
  end;

end.