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
  TensorFlow,
  RemObjects.Elements.System;

type
  TFObjectDisposedException = public class(Exception)
  public
    constructor (aObject: TFObject);
    begin
      inherited constructor($'{aObject.ToString} instance was already disposed.');
    end;
  end;

  TFObjectDisposeAction = public block(aObjectPtr: ^Void);

  TFObject = public abstract class(IDisposable)
  private
    fDisposed: Boolean := false;
    fObjectPtr: ^Void := nil;
    fDisposeAction: TFObjectDisposeAction;

    finalizer;
    begin
      if not fDisposed then 
        Dispose(false);
    end;

    method get_ObjectPtr: ^Void;
    begin
      if fDisposed then 
        raise new TFObjectDisposedException(self);
      exit fObjectPtr;
    end;
  protected
    constructor withObjectPtr(aPtr: ^Void) DisposeAction(aAction: TFObjectDisposeAction);
    begin
      fObjectPtr := aPtr;
      fDisposeAction := aAction;
    end;

    method Dispose(aDisposing: Boolean); virtual;
    begin
      if fDisposed then 
        exit;
     
      fDisposeAction(fObjectPtr);
      fDisposed := true;
    end;
  public
    method Dispose;
    begin
      Dispose(true);
    end;

    property ObjectPtr: ^Void read get_ObjectPtr;
  end;

  TFBuffer = class(TFObject)
  private
    fDisposeAction: TFObjectDisposeAction := method (aObjectPtr: ^Void) begin
      TF_DeleteBuffer(^TF_Buffer(aObjectPtr));
    end;

  public
    constructor;
    begin
      inherited constructor withObjectPtr(TF_NewBuffer()) DisposeAction(fDisposeAction);
    end;

    constructor(const aProtoBuf: array of Byte);
    begin
      inherited constructor withObjectPtr(TF_NewBufferFromString(aProtoBuf, aProtoBuf.Length))
        DisposeAction(fDisposeAction);
    end;
  end;

end.