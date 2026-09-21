"""Source-free tests of the scoring semantics, not outcome-dependent checks."""
import itertools
import unittest

import numpy as np
import pandas as pd

from norman_reference import (PATTERN_INDEX, PATTERNS, TUPLES, allocated_means,
                              fit_baselines, allocation_matrix, score_arrays)


class ReferenceScoringTests(unittest.TestCase):
    def test_balanced_roles(self):
        self.assertEqual([len(PATTERN_INDEX[p]) for p in PATTERNS], [3,6,6,6,6])
        self.assertEqual(sum(map(len,PATTERN_INDEX.values())),27)
        for pattern in PATTERNS:
            tuples=np.asarray([TUPLES[i] for i in PATTERN_INDEX[pattern]])
            for role in range(3):
                np.testing.assert_array_equal(np.bincount(tuples[:,role],minlength=3),
                                              np.full(3,len(tuples)//3))

    def test_full_algebra_matches_independent_scalar_oracle(self):
        rng=np.random.default_rng(411)
        b=rng.normal(size=(2,3,3,11))
        states=rng.normal(size=(2,3,3,3,11))
        states[:,:,0,:,:]=rng.normal(size=(2,3,1,11))
        states[:,:,1,:,:]=rng.normal(size=(2,3,1,11))
        target=rng.normal(size=11)
        scales=rng.uniform(.1,2.,size=11)
        atomic,utility,v,k,distance,audit=score_arrays(b,states,target,scales)
        for label,depth,model,assignment in itertools.product(range(2),range(3),range(4),range(27)):
            o,p,m=TUPLES[assignment]
            predicted=np.zeros(11) if model==3 else states[label,depth,model,m]-b[label,depth,p]
            observed=target-b[label,depth,o]
            expected=-sum(((predicted[g]-observed[g])/scales[g])**2 for g in range(11))/11
            self.assertAlmostEqual(atomic[label,depth,model,assignment],expected,places=11)
        self.assertLess(max(audit.values()),1e-11)
        np.testing.assert_allclose(k[:,:,:2],0.,atol=1e-13)
        np.testing.assert_array_equal(distance[:,:,:2],0.)
        self.assertGreater(np.max(np.abs(k[:,:,2])),.001)

    def test_same_state_model_pair_invariant_S_to_D_but_P_can_cross(self):
        # Conditioning-sensitive B wins under S, but has K>0 and loses more
        # under P, producing a crossing without changing identity or fitting.
        b=np.asarray([[[[-1.],[0.],[1.]]]])
        state=np.zeros((1,1,3,3,1))
        state[:,:,0,:,:]=.9
        state[:,:,1,:,:]=1.2
        state[:,:,2,:,:]=b
        _,u,_,_,_,_=score_arrays(b,state,np.zeros(1),np.ones(1))
        s=u[0,0,0,0]-u[0,0,2,0]
        d=u[0,0,0,4]-u[0,0,2,4]
        p=u[0,0,0,2]-u[0,0,2,2]
        self.assertAlmostEqual(s,d,places=14)
        self.assertLess(s*p,0)

    def test_nested_equal_gemgroup_allocation_means(self):
        ids=[f'cell{i}' for i in range(39)]
        groups=['g1']*18+['g2']*21
        depths=(1,3,6)
        rows=[]
        for label in range(2):
            for group in ('g1','g2'):
                indices=np.flatnonzero(np.asarray(groups)==group)
                indices=np.roll(indices,label)
                for block in range(3):
                    for rank,idx in enumerate(indices[block*6:(block+1)*6]):
                        rows.append((label,group,block,rank,idx,ids[idx]))
        manifest=pd.DataFrame(rows,columns=['assignment','gemgroup','block','within_block_index',
                                           'eval_control_index','cell_id'])
        matrix,n=allocation_matrix(ids,groups,manifest,depths,labels=2)
        values=np.arange(39*5).reshape(39,5).astype(float)
        means=allocated_means(matrix,values,n,depths,labels=2)
        cell_lookup={x:i for i,x in enumerate(ids)}
        for label,di,block in itertools.product(range(2),range(3),range(3)):
            selection=manifest[(manifest.assignment==label)&(manifest.block==block)&
                               (manifest.within_block_index<depths[di])]
            self.assertEqual(len(selection),2*depths[di])
            direct=np.mean([values[[cell_lookup[x] for x in selection[selection.gemgroup==g].cell_id]].mean(axis=0)
                            for g in ('g1','g2')],axis=0)
            np.testing.assert_allclose(means[label,di,block],direct)
        repeated,_=allocation_matrix(ids,groups,manifest,depths,labels=2)
        np.testing.assert_array_equal(matrix.toarray(),repeated.toarray())
        bad=manifest.copy()
        bad.loc[1,'eval_control_index']=bad.loc[0,'eval_control_index']
        with self.assertRaises(ValueError):
            allocation_matrix(ids,groups,bad,depths,labels=2)

    def test_ridge_unpenalized_intercept_and_training_only(self):
        train=['ctrl','A','B','C','A+B']
        means=np.array([[1.,2.],[2.,3.],[3.,5.],[4.,7.],[4.,6.]])
        result=fit_baselines(train,means,['B+C'])
        np.testing.assert_allclose(result['additive'],[[6.,10.]])
        self.assertLess(result['ridge_normal_equation_max_abs'],1e-12)
        shifted=fit_baselines(train,means+10.,['B+C'])
        np.testing.assert_allclose(shifted['compositional_ridge'],result['compositional_ridge']+10.)
        np.testing.assert_allclose(shifted['ridge_slope'],result['ridge_slope'])
        with self.assertRaises(ValueError):
            fit_baselines(train,means,['A+B'])
        with self.assertRaises(ValueError):
            fit_baselines(train,means,['B+D'])


if __name__=='__main__':
    unittest.main()
