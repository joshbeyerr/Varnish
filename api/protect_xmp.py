# protect_xmp.py
from libxmp import XMPFiles, XMPMeta, consts
from PIL import Image
import io, os

PLUS_NS = "http://ns.useplus.org/LDF/1.0/"
PLUS_PREFIX = "plus"
DMI_PROHIBITED = "http://ns.useplus.org/ldf/vocab/DMI-PROHIBITED"

def set_do_not_train_xmp(image_path: str) -> dict:
    """
    Try to embed XMP (with PLUS:DataMining=DMI-PROHIBITED) directly into the file.
    If embedding fails (format/codec limitations), write a sidecar .xmp.
    Returns a dict telling you what happened.
    """
    # 1) Ensure the image can be opened (also normalizes weird inputs)
    with Image.open(image_path) as im:
        # keep file as-is; we just verify it's an image
        im.verify()

    # 2) Prepare XMP
    xmp = XMPMeta()
    try:
        XMPMeta.register_namespace(PLUS_NS, PLUS_PREFIX)
    except Exception:
        pass  # already registered

    xmp.set_property(PLUS_NS, "DataMining", DMI_PROHIBITED)
    # Nice-to-have standard fields (non-required)
    # xmp.set_property(consts.XMP_NS_DC, "rights", "All rights reserved.")
    # xmp.set_property(consts.XMP_NS_DC, "creator", "Your Name")

    # 3) Try in-place embed (works well for JPEG/TIFF; spotty for PNG/WebP)
    try:
        xf = XMPFiles(file_path=image_path, open_forupdate=True)
        existing = xf.get_xmp()
        if existing is not None:
            # merge: existing wins on conflicts except DataMining which we overwrite
            for (ns, prop, _), val in existing.properties():
                if not (ns == PLUS_NS and prop == "DataMining"):
                    xmp.set_property(ns, prop, val)
        xf.put_xmp(xmp)
        xf.close_file()
        return {"embedded": True, "sidecar": None}
    except Exception as e:
        # 4) Fallback: sidecar .xmp (good enough for demo & provenance viewers)
        sidecar = image_path + ".xmp"
        with open(sidecar, "wb") as f:
            f.write(xmp.serialize_to_str(omit_packet_wrapper=False).encode("utf-8"))
        return {"embedded": False, "sidecar": sidecar, "error": str(e)}
