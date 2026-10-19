/* Restrict SDL's eight-device EGL enumeration to Mesa's software device.
 * Build on the pod: cc -shared -fPIC software_egl.c -o STATE/software_egl.so
 *   -Wl,--no-as-needed -l:libEGL.so.1 -ldl
 * SDL_VIDEO_EGL_DRIVER points here; other EGL symbols resolve via libEGL.
 */
#include <dlfcn.h>
#include <string.h>
typedef void *Device;
typedef unsigned int Boolean;
typedef void (*Proc)(void);
typedef Proc (*GetProc)(const char *);
static GetProc real_get(void) {
    static GetProc get;
    if (!get) get = (GetProc)dlsym(dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL), "eglGetProcAddress");
    return get;
}
static Boolean software_devices(int capacity, Device *out, int *count) {
    GetProc get = real_get();
    Boolean (*query)(int, Device *, int *) = (void *)get("eglQueryDevicesEXT");
    const char *(*string)(Device, int) = (void *)get("eglQueryDeviceStringEXT");
    Device devices[64]; int n = 0;
    if (!query(64, devices, &n)) return 0;
    *count = 0;
    for (int i = 0; i < n && i < 64; ++i) {
        const char *ext = string(devices[i], 0x3055);
        if (ext && strstr(ext, "EGL_MESA_device_software")) {
            if (capacity > 0 && out) out[0] = devices[i];
            *count = 1;
            return 1;
        }
    }
    return 1;
}
Proc eglGetProcAddress(const char *name) {
    if (!strcmp(name, "eglQueryDevicesEXT")) return (Proc)software_devices;
    return real_get()(name);
}
