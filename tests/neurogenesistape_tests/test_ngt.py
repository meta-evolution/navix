import sys
import neurogenesistape
print("Modules:", list(sys.modules.keys()))
print("ngt in modules:", "ngt" in sys.modules)
print("neurogenesistape in modules:", "neurogenesistape" in sys.modules)

try:
    import ngt
    print("Successfully imported ngt")
    print("ngt.__version__:", ngt.__version__)
except ImportError as e:
    print("Failed to import ngt:", e)
