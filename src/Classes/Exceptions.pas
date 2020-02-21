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
  RemObjects.Elements.System,
  TensorFlow.Island.Api;

type
  BufferFileNotExistException = public class(Exception)
  public
    constructor(aFile: not nullable String);
    begin
      inherited constructor($'Buffer file {aFile} not existing.');
    end;
  end;

  DeviceNameAlreadySetException = public class(Exception)
  public
    constructor withExistingName(aName: String);
    begin
      inherited constructor($'Device name already set, existing name {aName}.');
    end;
  end;

  DeviceNameEmptyException = public class(Exception)
  public
    constructor;
    begin
      inherited constructor($'Trying to set an empty device name.');
    end;
  end;

  TensorDataSizeException = public class(Exception)
  public
    constructor withTensorDataSize(aTensorDataSize: Integer) ShapeSize(aShapeSize: Integer);
    begin
      inherited constructor(
        $'Tensor data[size={aTensorDataSize}] inconsistent ' +
        $'with shape[size={aShapeSize}].');
    end;
  end;

  OsBitSizeException = public class(Exception)
  public
    constructor withDetectedOsBitSize(aSize: Integer);
    begin
      inherited constructor($'Invalid OS bit size {aSize}. Support 64bit only.');
    end;
  end;

  LibraryLoadException = public class(Exception)
  public
    constructor withLibName(aName: NotNull<String>) Error(aMsg: NotNull<String>);
    begin
      inherited constructor($'Cannot load TensorFlow library "{aName}". {aMsg}');
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
    constructor withOpType(aOpType: not nullable String) Error(aErr: NotNull<String> := '');
    begin
      inherited constructor($'Fail creating Op[type={aOpType}]. {aErr}');
    end;
  end;

  PartialRunSetupException = public class(Exception)
  public
    constructor withError(aErr: String);
    begin
      inherited constructor($'Fail setting up partial run with error "{aErr}".');
    end;
  end;

  PartialRunTokenNotSetupException = public class(Exception)
  public
    constructor;
    begin
      inherited constructor('Cannot invoke PartialRun with uninitialized token.');
    end;
  end;

  SessionCreateException = public class(Exception)
  public
    constructor withError(aErr: NotNull<String>);
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
      inherited constructor($'Fail encoding "{aValue}". {aErrMsg}');
    end;
  end;

  TensorCreateException = class(Exception)
  public
    constructor withTensorType(aType: DataType);
    begin
      var msg := $'Fail creating tensor[type={aType.ToString}].';
      inherited constructor(msg);
    end;
  end;

end.