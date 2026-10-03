use std::collections::HashMap;

use anyhow::{Context, Result};

type Node = (u32, u32);
pub type ClusterIds = HashMap<Node, u32>;

/// Assign IDs in Python's order: by the smallest (file, document) in each component.
///
/// The completed union-find may have chosen any member as its root. Only component
/// minima are sorted, keeping auxiliary memory proportional to the cluster count.
pub fn build_cluster_ids(parents: &HashMap<Node, Node>) -> Result<ClusterIds> {
    let mut minima = HashMap::<Node, Node>::new();
    for &node in parents.keys() {
        let mut root = node;
        loop {
            let parent = parents[&root];
            if parent == root {
                break;
            }
            root = parent;
        }
        minima
            .entry(root)
            .and_modify(|minimum| *minimum = (*minimum).min(node))
            .or_insert(node);
    }

    let mut roots: Vec<_> = minima.into_iter().collect();
    roots.sort_unstable_by_key(|&(_, minimum)| minimum);
    roots
        .into_iter()
        .enumerate()
        .map(|(id, (root, _))| Ok((root, checked_cluster_id(id)?)))
        .collect()
}

fn checked_cluster_id(id: usize) -> Result<u32> {
    u32::try_from(id).context("Too many MinHash clusters for u32 cluster IDs")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        assert!(build_cluster_ids(&HashMap::new()).unwrap().is_empty());
    }

    #[test]
    fn ids_follow_minimum_members_instead_of_roots() {
        let parents = HashMap::from([
            ((0, 0), (1, 0)),
            ((1, 0), (2, 2)),
            ((2, 2), (2, 2)),
            ((0, 2), (1, 1)),
            ((1, 1), (1, 1)),
            ((2, 0), (1, 1)),
            ((3, 0), (3, 0)),
        ]);
        let original = parents.clone();
        assert_eq!(
            build_cluster_ids(&parents).unwrap(),
            HashMap::from([((2, 2), 0), ((1, 1), 1), ((3, 0), 2)])
        );
        assert_eq!(parents, original);

        let other_roots = HashMap::from([
            ((3, 0), (3, 0)),
            ((2, 0), (0, 2)),
            ((1, 1), (0, 2)),
            ((0, 2), (0, 2)),
            ((2, 2), (0, 0)),
            ((1, 0), (0, 0)),
            ((0, 0), (0, 0)),
        ]);
        assert_eq!(
            build_cluster_ids(&other_roots).unwrap(),
            HashMap::from([((0, 0), 0), ((0, 2), 1), ((3, 0), 2)])
        );
    }

    #[test]
    fn sentinel_component_is_ordered_by_its_real_members() {
        let sentinel = (u32::MAX, u32::MAX);
        let parents = HashMap::from([
            (sentinel, sentinel),
            ((0, 1), sentinel),
            ((1, 2), (0, 1)),
            ((0, 2), (0, 2)),
        ]);
        assert_eq!(
            build_cluster_ids(&parents).unwrap(),
            HashMap::from([(sentinel, 0), ((0, 2), 1)])
        );
    }

    #[test]
    fn cluster_id_limit() {
        assert_eq!(checked_cluster_id(u32::MAX as usize).unwrap(), u32::MAX);
        #[cfg(target_pointer_width = "64")]
        assert!(checked_cluster_id(u32::MAX as usize + 1).is_err());
    }
}
