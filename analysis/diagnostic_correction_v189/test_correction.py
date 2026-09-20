"""Scientific regression checks for algebraic-zero handling, not model fitting."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
SPEC=importlib.util.spec_from_file_location('diagnostic_correction_test_target',Path(__file__).with_name('replay.py'))
TARGET=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TARGET)
diagnose,structural=TARGET.diagnose,TARGET.structural

class StructuralZeroTests(unittest.TestCase):
    def test_design_identity_does_not_use_effect_magnitude(self):
        row=dict(source_scheme_id='NAIVE_SHARED',target_scheme_id='UNIT_AWARE_CROSSFIT',
            method_a='NO_CHANGE_DIRECT_V1',method_b='RBF_KERNEL_RIDGE_DIRECT_V1')
        for value in (0.,1e-300,1e6):
            self.assertTrue(structural(dict(row,root_paired_change=value)))
        self.assertFalse(structural(dict(row,target_scheme_id='INDEPENDENT_SPLIT')))
        self.assertFalse(structural(dict(row,method_b='SCGEN_2_1_1_ABSOLUTE_V1')))

    def test_affine_direct_pair_identity(self):
        # Three equal-depth blocks of reference cells. Exact integer arithmetic
        # checks the squared-loss pair identity before taking averages.
        a=np.array([1,2,-3],dtype=np.int64);b=np.array([-2,4,1],dtype=np.int64)
        observed=np.array([[2,7,3],[-1,5,8],[5,0,1]],dtype=np.int64)
        contrasts=(-np.sum((observed-a)**2,axis=1)+np.sum((observed-b)**2,axis=1))
        self.assertTrue(np.array_equal(contrasts,2*observed@(a-b)+(b@b-a@a)))
        # Multiply by nine to avoid rational arithmetic when comparing with
        # the shared mean of the three observation targets.
        union=observed.sum(axis=0)
        shared9=-np.sum((union-3*a)**2)+np.sum((union-3*b)**2)
        self.assertEqual(int(shared9),3*int(contrasts.sum()))

    def test_nonstructural_degeneracy_keeps_whole_family_fallback(self):
        x=np.random.default_rng(23).normal(size=(8,56));mask=np.zeros(56,dtype=bool)
        mask[-10:]=True;x[:,mask]=0
        good=diagnose(x,mask)
        self.assertFalse(good['fallback'])
        self.assertTrue(np.all(good['statistics'][:,mask]==0))
        self.assertTrue(np.all(good['lower'][mask]==0))
        x[:,0]=4
        bad=diagnose(x,mask)
        self.assertTrue(bad['fallback'])
        self.assertTrue(np.all(np.isneginf(bad['lower'])))
        self.assertTrue(np.all(np.isposinf(bad['upper'])))

if __name__=='__main__':unittest.main()
