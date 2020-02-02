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
  TensorFlow;

type
  BufferFileNotExistException = public class(Exception)
  public
    constructor (aFile: not nullable String);
    begin
      inherited constructor($'Buffer file {aFile} not existing.');
    end;
  end;

  InvalidRectangularTensorData = public class(Exception)
  public
    constructor(aMsg: not nullable String);
    begin
      inherited constructor(aMsg);
    end;
  end;

  InvalidShapeDimIndexException = public class(Exception)
  public
    constructor (aIndex: Int32; aNumDims: Int32);
    begin
      var msg := $'Invalid dim index {aIndex}, NumDims = {aNumDims}.';
      inherited constructor(msg);
    end;
  end;

  ObjectDisposedException = public class(Exception)
  public
    constructor (aObject: DisposableObject);
    begin
      inherited constructor($'{aObject.GetType.Name} instance already disposed.');
    end;
  end;

  OpCreateException = public class(Exception)
  public
    constructor withOpType(aOpType: not nullable String)
      Message(aMsg: not nullable String := '');
    begin
      inherited constructor($'Fail creating {aOpType}. {aMsg}');
    end;
  end;

  SessionCreateException = public class(Exception)
  public
    constructor withMessage(aMsg: not nullable String);
    begin
      inherited constructor(aMsg);
    end;
  end;

  TensorCreateException = class(Exception)
  public
    constructor(aType: TF_DataType);
    begin
      var typeStr := Helper.TFDataTypeToString(aType);
      var msg := $'Cannot create tensor for type {typeStr}';
      inherited constructor(msg);
    end;
  end;

  UnSupportedTypeException = public class(Exception)
  public
    constructor(aType: &Type);
    begin
      var msg := $'Cannot convert {aType.ToString} to TensorFlow DataType.';
      inherited constructor(msg);
    end;
  end;

end.