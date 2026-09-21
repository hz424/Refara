"""Independent small examples of the effect/state comparison geometry."""
import itertools
import unittest

import numpy as np
import pandas as pd

from score_direct_effect import (
    PATTERNS, TUPLES, annotate, effect_utilities, make_tables, pattern, reference_distance,
)


def scalar_state_scores(states, controls, treated, scales):
    atomic = []
    for o, p, m in itertools.product(range(3), repeat=3):
        residual = (states[m] - treated + controls[o] - controls[p]) / scales
        atomic.append(-sum(float(x*x) for x in residual) / len(residual))
    return np.array([np.mean([atomic[i] for i,t in enumerate(TUPLES) if pattern(t)==p])
                     for p in PATTERNS])


class GeometryTests(unittest.TestCase):
    def test_fixed_effect_atomic_values_and_balanced_patterns(self):
        b = np.array([[-1.,2.], [0.,-2.], [1.,3.]])
        f, y, s = np.array([.4, -.5]), np.array([1., .2]), np.array([2., 3.])
        atomic, utility = effect_utilities(f,y,b,s)
        for i,(o,_,_) in enumerate(TUPLES):
            expected = -sum(((f[g]-y[g]+b[o,g])/s[g])**2 for g in range(2))/2
            self.assertAlmostEqual(atomic[i], expected, places=14)
        np.testing.assert_allclose(utility, np.repeat(utility[0],5), atol=1e-14, rtol=0)

    def test_crossing_and_both_stable_directions(self):
        b = np.array([[-1.], [0.], [1.]])
        v = reference_distance(b,np.ones(1))
        self.assertAlmostEqual(v,2.)
        results = []
        for f, state in [(0.,0.), (0.,2.), (4.,0.)]:
            _, e = effect_utilities(np.array([f]),np.zeros(1),b,np.ones(1))
            u = scalar_state_scores(np.full((3,1),state),b,np.zeros(1),np.ones(1))
            results.append(dict(d_S=e[0]-u[0],V=v,d_D=e[4]-u[4]))
        got=annotate(pd.DataFrame(results))
        self.assertEqual(got.classification.tolist(),
                         ['state_to_effect','effect_preferred_both','state_preferred_both'])

    def test_conditioning_sensitive_state_and_zero_distance(self):
        b=np.array([[-1.,2.],[0.,-2.],[1.,3.]])
        states=np.array([[2.,4.],[-3.,2.],[1.,-4.]])
        s=np.array([2.,3.]); y=np.array([.7,-.2])
        u=scalar_state_scores(states,b,y,s)
        self.assertAlmostEqual(u[4]-u[0],-reference_distance(b,s),places=14)
        self.assertGreater(abs(u[2]-u[3]),1e-3)
        zeros=np.zeros_like(b)
        u=scalar_state_scores(states,zeros,y,s)
        self.assertEqual(reference_distance(zeros,s),0.)
        self.assertAlmostEqual(u[4],u[0],places=14)

    def test_boundaries_are_ties(self):
        a=annotate(pd.DataFrame([dict(d_S=0.,V=2.,d_D=2.),
                                 dict(d_S=-2.,V=2.,d_D=0.),
                                 dict(d_S=-1.,V=0.,d_D=-1.)]))
        self.assertEqual(a.classification.tolist(),['numerical_tie','numerical_tie','state_preferred_both'])
        self.assertFalse(a.observed_crossing.any())

    def test_aggregation_preserves_v_and_conditions(self):
        e=np.zeros((2,2,2,5)); state=np.zeros((2,2,2,3,5)); v=np.array([[1.,2.],[3.,4.]])
        state[...,0]=.5
        state[...,4]=.5-v[...,None]
        tables,_,_=make_tables(e,state,v,['A+B','C+D'],[8,16])
        overall=tables['overall_mean_pairs']
        np.testing.assert_array_equal(overall.V, [2.,2.,2.,3.,3.,3.])
        self.assertTrue(overall.observed_crossing.all())
        self.assertEqual(len(tables['task_allocation_pairs']),24)


if __name__=='__main__':
    unittest.main()
