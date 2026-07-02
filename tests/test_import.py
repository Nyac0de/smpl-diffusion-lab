def test_import_package():
    import smpl_diffusion_lab

    assert smpl_diffusion_lab is not None


def test_import_subpackages():
    import smpl_diffusion_lab.diffusion
    import smpl_diffusion_lab.pose
    import smpl_diffusion_lab.smpl
    import smpl_diffusion_lab.models
    import smpl_diffusion_lab.utils

    assert smpl_diffusion_lab.diffusion is not None
    assert smpl_diffusion_lab.pose is not None
    assert smpl_diffusion_lab.smpl is not None
    assert smpl_diffusion_lab.models is not None
    assert smpl_diffusion_lab.utils is not None
