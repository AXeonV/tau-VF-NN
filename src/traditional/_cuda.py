"""In-memory CUDA compilation for the two dense quadrature matrices."""
import ctypes as C
from pathlib import Path
import sys


SOURCE = r'''
extern "C" __global__ void quadrature(
    const double* z, const double* normal, const double* speed,
    const double* curvature, const double* chord, double* kernel,
    double* smooth, int n) {
    int entry = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (entry >= n*n) return;
    int row = entry/n, column = entry-row*n;
    if (row == column) {
        kernel[entry] = -curvature[row]*speed[row]/(2*n);
        smooth[entry] = log(speed[row]);
        return;
    }
    double dx = z[2*row]-z[2*column];
    double dy = z[2*row+1]-z[2*column+1];
    double distance2 = dx*dx+dy*dy;
    int offset = row-column;
    if (offset < 0) offset += n;
    kernel[entry] = (dx*normal[2*column]+dy*normal[2*column+1])
                    * (speed[column]/n)/distance2;
    smooth[entry] = .5*log(distance2)-chord[offset];
}
'''


def _check(code, operation):
    if code:
        raise RuntimeError(f'{operation} returned CUDA status {code}')


class QuadratureKernel:
    def __init__(self, torch):
        self.torch = torch
        if sys.platform == 'win32':
            lib = Path(torch.__file__).parent/'lib'
            nvrtc_path = next(lib.glob('nvrtc64_*_0.dll'), None)
            if nvrtc_path is None:
                raise OSError('NVRTC was not bundled with PyTorch')
            rtc = C.CDLL(str(nvrtc_path))
            self.driver = C.WinDLL('nvcuda.dll')
        else:
            rtc = C.CDLL('libnvrtc.so.12')
            self.driver = C.CDLL('libcuda.so.1')
        program = C.c_void_p()
        _check(rtc.nvrtcCreateProgram(C.byref(program), SOURCE.encode(), b'quadrature.cu', 0, None, None),
               'nvrtcCreateProgram')
        try:
            major, minor = torch.cuda.get_device_capability()
            options = (C.c_char_p*1)(f'--gpu-architecture=compute_{major}{minor}'.encode())
            status = rtc.nvrtcCompileProgram(program, len(options), options)
            if status:
                length = C.c_size_t()
                rtc.nvrtcGetProgramLogSize(program, C.byref(length))
                log = C.create_string_buffer(length.value)
                rtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(log.value.decode(errors='replace'))
            length = C.c_size_t()
            _check(rtc.nvrtcGetPTXSize(program, C.byref(length)), 'nvrtcGetPTXSize')
            ptx = C.create_string_buffer(length.value)
            _check(rtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
        finally:
            rtc.nvrtcDestroyProgram(C.byref(program))
        self.module = C.c_void_p()
        _check(self.driver.cuModuleLoadData(C.byref(self.module), ptx), 'cuModuleLoadData')
        self.kernel = C.c_void_p()
        _check(self.driver.cuModuleGetFunction(C.byref(self.kernel), self.module, b'quadrature'),
               'cuModuleGetFunction')

    def __call__(self, z, normal, speed, curvature, chord):
        n = len(z)
        kernel = speed.new_empty((n, n))
        smooth = speed.new_empty((n, n))
        values = [C.c_void_p(value.data_ptr()) for value in (z, normal, speed, curvature, chord, kernel, smooth)]
        values.append(C.c_int(n))
        arguments = (C.c_void_p*len(values))(*(C.addressof(value) for value in values))
        stream = C.c_void_p(self.torch.cuda.current_stream().cuda_stream)
        _check(self.driver.cuLaunchKernel(self.kernel, (n*n+255)//256, 1, 1, 256, 1, 1,
                                          0, stream, arguments, None), 'cuLaunchKernel')
        return kernel, smooth
