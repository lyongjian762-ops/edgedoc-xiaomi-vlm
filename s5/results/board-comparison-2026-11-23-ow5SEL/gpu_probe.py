"""Verify board GPU access with OpenCL and enumerate configured Vulkan devices."""

from contextlib import ExitStack
import ctypes as c
import json
from pathlib import Path
import struct

pointer, uint, int32, flags, size = c.c_void_p, c.c_uint32, c.c_int32, c.c_uint64, c.c_size_t
status_out = c.POINTER(int32)
objects = c.POINTER(pointer)


def function(library, name, result, *arguments):
    call = getattr(library, name)
    call.restype = result
    call.argtypes = arguments
    return call


def require_ok(stage, status):
    if status != 0:
        raise RuntimeError(f"{stage} returned {status}")


def opencl_probe():
    lib = c.CDLL("libOpenCL.so.1")
    platform_ids = function(lib, "clGetPlatformIDs", int32, uint, objects, c.POINTER(uint))
    device_ids = function(lib, "clGetDeviceIDs", int32, pointer, flags, uint, objects, c.POINTER(uint))
    device_info = function(lib, "clGetDeviceInfo", int32, pointer, uint, size, pointer, c.POINTER(size))
    create_context = function(lib, "clCreateContext", pointer, pointer, uint, objects, pointer, pointer, status_out)
    create_queue = function(lib, "clCreateCommandQueue", pointer, pointer, pointer, flags, status_out)
    create_program = function(lib, "clCreateProgramWithSource", pointer, pointer, uint, c.POINTER(c.c_char_p), c.POINTER(size), status_out)
    build = function(lib, "clBuildProgram", int32, pointer, uint, objects, c.c_char_p, pointer, pointer)
    create_kernel = function(lib, "clCreateKernel", pointer, pointer, c.c_char_p, status_out)
    create_buffer = function(lib, "clCreateBuffer", pointer, pointer, flags, size, pointer, status_out)
    set_arg = function(lib, "clSetKernelArg", int32, pointer, uint, size, pointer)
    dispatch = function(lib, "clEnqueueNDRangeKernel", int32, pointer, pointer, uint, c.POINTER(size), c.POINTER(size), c.POINTER(size), uint, objects, objects)
    read = function(lib, "clEnqueueReadBuffer", int32, pointer, pointer, uint, size, size, pointer, uint, objects, objects)

    platform = pointer()
    require_ok("platform", platform_ids(1, c.byref(platform), None))
    device = pointer()
    require_ok("GPU device", device_ids(platform, 1 << 2, 1, c.byref(device), None))

    def descriptor(key):
        length = size()
        require_ok("device info size", device_info(device, key, 0, None, c.byref(length)))
        result = c.create_string_buffer(length.value)
        require_ok("device info", device_info(device, key, length.value, result, None))
        return result.value.decode()

    with ExitStack() as cleanup:
        result = int32()

        def create(call, resource, *args):
            handle = call(*args, c.byref(result))
            require_ok(call.__name__, result.value)
            cleanup.callback(function(lib, "clRelease" + resource, int32, pointer), handle)
            return handle

        ctx = create(create_context, "Context", None, 1, c.byref(device), None, None)
        queue = create(create_queue, "CommandQueue", ctx, device, 0)
        source = c.c_char_p(b"__kernel void add(__global float *a, __global float *b, "
                            b"__global float *out) { size_t i = get_global_id(0); out[i] = a[i] + b[i]; }")
        program = create(create_program, "Program", ctx, 1, c.byref(source), None)
        require_ok("build kernel", build(program, 1, c.byref(device), b"-cl-std=CL1.2", None, None))
        kernel = create(create_kernel, "Kernel", program, b"add")
        count = 1024
        values = c.c_float * count
        a = values(*range(count))
        b = values(*(2 * i + 1 for i in range(count)))
        out = values()
        byte_count = c.sizeof(out)
        inputs = [create(create_buffer, "MemObject", ctx, (1 << 2) | (1 << 5), byte_count, data) for data in (a, b)]
        target = create(create_buffer, "MemObject", ctx, 1 << 1, byte_count, None)
        for index, value in enumerate([*inputs, target]):
            require_ok("set kernel argument", set_arg(kernel, index, c.sizeof(pointer), c.byref(pointer(value))))
        require_ok("dispatch", dispatch(queue, kernel, 1, None, c.byref(size(count)), None, 0, None, None))
        require_ok("read output", read(queue, target, 1, 0, byte_count, out, 0, None, None))
        error = max(abs(out[i] - (3 * i + 1)) for i in range(count))
        if error != 0:
            raise RuntimeError(f"vector add differed by {error}")
    return {"device": descriptor(0x102B), "version": descriptor(0x102F),
            "fp16_extension": "cl_khr_fp16" in descriptor(0x1030).split(),
            "vector_add_values": count, "max_abs_error": error}


def vulkan_probe():
    class InstanceInfo(c.Structure):
        _fields_ = [("sType", uint), ("pNext", pointer), ("flags", uint), ("app", pointer),
                    ("layer_count", uint), ("layers", pointer), ("extension_count", uint), ("extensions", pointer)]

    paths = sorted(str(p) for root in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d") for p in Path(root).glob("*.json"))
    lib = c.CDLL("libvulkan.so.1")
    make = function(lib, "vkCreateInstance", int32, c.POINTER(InstanceInfo), pointer, objects)
    instance = pointer()
    status = make(c.byref(InstanceInfo(1)), None, c.byref(instance))
    report = {"system_icd_files": paths, "create_instance_status": status, "devices": []}
    if status != 0:
        return report
    destroy = function(lib, "vkDestroyInstance", None, pointer, pointer)
    enum = function(lib, "vkEnumeratePhysicalDevices", int32, pointer, c.POINTER(uint), objects)
    properties = function(lib, "vkGetPhysicalDeviceProperties", None, pointer, pointer)
    try:
        count = uint()
        require_ok("device count", enum(instance, c.byref(count), None))
        devices = (pointer * count.value)()
        require_ok("device list", enum(instance, c.byref(count), devices))
        for device in devices:
            data = c.create_string_buffer(4096)
            properties(device, data)
            _, _, vendor, product, kind = struct.unpack_from("=5I", data)
            report["devices"].append({"name": data.raw[20:276].split(b"\x00", 1)[0].decode(),
                                      "vendor_id": vendor, "device_id": product, "device_type": kind})
    finally:
        destroy(instance, None)
    return report


if __name__ == "__main__":
    print(json.dumps({"opencl": opencl_probe(), "vulkan": vulkan_probe()}, indent=2))
