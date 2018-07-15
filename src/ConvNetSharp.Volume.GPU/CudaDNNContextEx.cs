//using System;
//using System.Diagnostics;
//using ManagedCuda;
//using ManagedCuda.BasicTypes;
//using ManagedCuda.CudaDNN;

//namespace ConvNetSharp.Volume.GPU
//{
//    /// <summary>
//    /// Temporary class (until it is integrated in ManagedCuda)
//    /// </summary>
//    public class CudaDNNContextEx : CudaDNNContext
//    {
//        /// <summary>
//        ///     This function performs backward dropout operation over dy returning results in dx. If during
//        ///     forward dropout operation value from x was propagated to y then during backward operation value
//        ///     from dy will be propagated to dx, otherwise, dx value will be set to 0.
//        /// </summary>
//        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>
//        /// <param name="dyDesc">Handle to a previously initialized tensor descriptor.</param>
//        /// <param name="dy">Pointer to data of the tensor described by the dyDesc descriptor.</param>
//        /// <param name="dxDesc">Handle to a previously initialized tensor descriptor.</param>
//        /// <param name="dx">Pointer to data of the tensor described by the dxDesc descriptor.</param>
//        /// <param name="reserveSpace">
//        ///     Data pointer to GPU memory used by this function. It is expected that contents of
//        ///     reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.
//        /// </param>
//        public void DropoutBackward(DropoutDescriptor dropoutDesc,
//            TensorDescriptor dyDesc,
//            CudaDeviceVariable<double> dy,
//            TensorDescriptor dxDesc,
//            CudaDeviceVariable<double> dx,
//            CudaDeviceVariable<byte> reserveSpace)
//        {
//            var res = CudaDNNNativeMethods.cudnnDropoutBackward(this.Handle, dropoutDesc.Desc, dyDesc.Desc, dy.DevicePointer, dxDesc.Desc, dx.DevicePointer,
//                reserveSpace.DevicePointer, reserveSpace.SizeInBytes);
//            Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutBackward", res);
//            if (res != cudnnStatus.Success)
//            {
//                throw new CudaDNNException(res);
//            }
//        }

//        /// <summary>
//        ///     This function performs backward dropout operation over dy returning results in dx. If during
//        ///     forward dropout operation value from x was propagated to y then during backward operation value
//        ///     from dy will be propagated to dx, otherwise, dx value will be set to 0.
//        /// </summary>
//        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>
//        /// <param name="dyDesc">Handle to a previously initialized tensor descriptor.</param>
//        /// <param name="dy">Pointer to data of the tensor described by the dyDesc descriptor.</param>
//        /// <param name="dxDesc">Handle to a previously initialized tensor descriptor.</param>
//        /// <param name="dx">Pointer to data of the tensor described by the dxDesc descriptor.</param>
//        /// <param name="reserveSpace">
//        ///     Data pointer to GPU memory used by this function. It is expected that contents of
//        ///     reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.
//        /// </param>
//        public void DropoutBackward(DropoutDescriptor dropoutDesc,
//            TensorDescriptor dyDesc,
//            CudaDeviceVariable<float> dy,
//            TensorDescriptor dxDesc,
//            CudaDeviceVariable<float> dx,
//            CudaDeviceVariable<byte> reserveSpace)
//        {
//            var res = CudaDNNNativeMethods.cudnnDropoutBackward(this.Handle, dropoutDesc.Desc, dyDesc.Desc, dy.DevicePointer, dxDesc.Desc, dx.DevicePointer,
//                reserveSpace.DevicePointer, reserveSpace.SizeInBytes);
//            Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutBackward", res);
//            if (res != cudnnStatus.Success)
//            {
//                throw new CudaDNNException(res);
//            }
//        }

//        /// <summary>
//        ///     This function performs forward dropout operation over x returning results in y. If dropout was
//        ///     used as a parameter to cudnnSetDropoutDescriptor, the approximately dropout fraction of x values
//        ///     will be replaces by 0, and the rest will be scaled by 1/(1-dropout) This function should not be
//        ///     running concurrently with another cudnnDropoutForward function using the same states.
//        /// </summary>
//        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>
//        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
//        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
//        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
//        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
//        /// <param name="reserveSpace">
//        ///     Data pointer to GPU memory used by this function. It is expected that contents of
//        ///     reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.
//        /// </param>
//        public void DropoutForward(DropoutDescriptor dropoutDesc,
//            TensorDescriptor xDesc,
//            CudaDeviceVariable<double> x,
//            TensorDescriptor yDesc,
//            CudaDeviceVariable<double> y,
//            CudaDeviceVariable<byte> reserveSpace)
//        {
//            var res = CudaDNNNativeMethods.cudnnDropoutForward(this.Handle, dropoutDesc.Desc, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer, reserveSpace.DevicePointer,
//                reserveSpace.SizeInBytes);
//            Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutForward", res);
//            if (res != cudnnStatus.Success)
//            {
//                throw new CudaDNNException(res);
//            }
//        }

//        /// <summary>
//        ///     This function performs forward dropout operation over x returning results in y. If dropout was
//        ///     used as a parameter to cudnnSetDropoutDescriptor, the approximately dropout fraction of x values
//        ///     will be replaces by 0, and the rest will be scaled by 1/(1-dropout) This function should not be
//        ///     running concurrently with another cudnnDropoutForward function using the same states.
//        /// </summary>
//        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>
//        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
//        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
//        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
//        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
//        /// <param name="reserveSpace">
//        ///     Data pointer to GPU memory used by this function. It is expected that contents of
//        ///     reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.
//        /// </param>
//        public void DropoutForward(DropoutDescriptor dropoutDesc,
//            TensorDescriptor xDesc,
//            CudaDeviceVariable<float> x,
//            TensorDescriptor yDesc,
//            CudaDeviceVariable<float> y,
//            CudaDeviceVariable<byte> reserveSpace)
//        {
//            var res = CudaDNNNativeMethods.cudnnDropoutForward(this.Handle, dropoutDesc.Desc, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer, reserveSpace.DevicePointer,
//                reserveSpace.SizeInBytes);
//            if (res != cudnnStatus.Success)
//            {
//                throw new CudaDNNException(res);
//            }
//        }

//        /// <summary>
//        ///     This function is used to query the amount of reserve needed to run dropout with the input dimensions given by
//        ///     xDesc.
//        ///     The same reserve space is expected to be passed to cudnnDropoutForward and cudnnDropoutBackward, and its contents
//        ///     is
//        ///     expected to remain unchanged between cudnnDropoutForward and cudnnDropoutBackward calls.
//        /// </summary>
//        /// <param name="xDesc">Handle to a previously initialized tensor descriptor, describing input to a dropout operation.</param>
//        public SizeT GetDropoutReserveSpaceSize(TensorDescriptor xDesc)
//        {
//            var sizeInBytes = new SizeT();
//            var res = CudaDNNNativeMethods.cudnnDropoutGetReserveSpaceSize(xDesc.Desc, ref sizeInBytes);
            
//            if (res != cudnnStatus.Success)
//            {
//                throw new CudaDNNException(res);
//            }
//            return sizeInBytes;
//        }

//        /// <summary>
//        ///     This function is used to query the amount of space required to store the states of the random number generators
//        ///     used by cudnnDropoutForward function.
//        /// </summary>
//        public SizeT GetDropoutStateSize()
//        {
//            var sizeInBytes = new SizeT();
//            var res = CudaDNNNativeMethods.cudnnDropoutGetStatesSize(this.Handle, ref sizeInBytes);
//            if (res != cudnnStatus.Success)
//            {
//                throw new CudaDNNException(res);
//            }
//            return sizeInBytes;
//        }
//    }
//}