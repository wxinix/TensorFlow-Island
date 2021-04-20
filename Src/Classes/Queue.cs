/*
MIT License Copyright(c) 2019-2020 Wuping Xin.

Permission is hereby granted, free of charge, to  any  person obtaining a copy
of this software and associated documentation files (the "Software"), to  deal
in the Software  without restriction,  including without limitation the rights
to use, copy, modify,  merge,  publish,  distribute, sublicense,  and/or  sell
copies   of   the  Software, and  to permit persons to  whom  the Software  is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
IMPLIED, INCLUDING BUT  NOT  LIMITED TO THE   WARRANTIES OF  MERCHANTABILITY,
FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO  EVENT SHALL THE
AUTHORS  OR COPYRIGHT  HOLDERS BE  LIABLE  FOR  ANY CLAIM,  DAMAGES OR   OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is the driver for the operation generator, using the information provided
by the Tensorflow run-time to produce strongly-typed and high level methods on
the TensorFlow.Island.Graph class.  The  output is a  partial class that is in
line with TensorFlow.Island library.
 */

using RemObjects.Elements.System;

namespace TensorFlow.Island.Classes
{
    /// <summary>
    ///   Base class for queue implementations. Python implementation
    ///   ../../../TensorFlow/r1.15.0/tensorflow/python/ops/data_flow_ops.py
    /// </summary>
    public abstract class QueueBase
    {
        public QueueBase(Session! session) => Session = session;

        public abstract Operation Enqueue(Output[] components, long? timeout_ms = null, string opName = null);
        public abstract Output[] Dequeue(long? timeout_ms = null, string opName = null);
        public abstract Output GetSize(string opName = null);

        protected Session Session { get; private set; }
    }

    /// <summary>
    ///   A first-in-first-out queue that supports batching variable-sized tensors by padding.
    /// </summary>
    public class PaddingFIFOQueue : QueueBase
    {
        private Output _handle;
        private DataType [] _componentTypes;

        public PaddingFIFOQueue(
            Session! session,
            DataType[]! componentTypes,
            Shape[]! shapes,
            int? capacity = null,
            string container = null,
            string opName = null
            ) : base (session)
        {
            _componentTypes = componentTypes;
            _handle = Session.Graph.PaddingFIFOQueueV2(componentTypes, shapes, capacity, container, opName);
        }

        public override Operation Enqueue(Output[] components, long? timeout_ms = null, string opName = null)
        {
            return Session.Graph.QueueEnqueueV2(_handle, components, timeout_ms, opName);
        }

        public Tensor[] EnqueueExecute(Output[] components, Tensor[] inputValues, long? timeout_ms = null, string opName = null)
        {
            Operation enqueueOp = Enqueue(components, timeout_ms, opName);
            return Session.Runner.AddInputs(components, inputValues).AddTarget(enqueueOp).Run()?.ToArray();
        }

        public override Output[] Dequeue(long? timeout_ms = null, string opName = null)
        {
            return Session.Graph.QueueDequeueV2(_handle, _componentTypes, timeout_ms, opName);
        }

        public Tensor[] DequeueExecute(long? timeout_ms = null, string opName = null)
        {
            Output[] values = Session.Graph.QueueDequeueV2 (_handle, _componentTypes, timeout_ms, opName);
            return Session.Runner.Run(values)?.ToArray();
        }

        public T[] DequeueExecute<T>(long? timeout_ms = null, string opName = null)
        {
            var vals = DequeueExecute(timeout_ms, opName).Select(x => x.AsScalar<T>().Item2).ToArray();
            T[] result = new T[vals.Length];
            
            for (int i = 0; i < vals.Length - 1; i++) {
                result[i] = vals[i];
            }
            
            return result;
        }

        /// <param name="opName">If not specified, 'QueueSizeV2'. Otherwise, specified.</param>
        public override Output GetSize(string opName = null)
        {
            return Session.Graph.QueueSizeV2(_handle, opName);
        }

        public int? GetSizeExecute(string opName = null)
        {
            Output sizeOutput = GetSize(opName);
            return Session.Runner.Run(sizeOutput)?.AsScalar<int>().Item2;
        }
    }
}