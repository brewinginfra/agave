use {
    agave_random::weighted::WeightedU64Index,
    rand::Rng,
    rand_chacha::{rand_core::SeedableRng, ChaChaRng},
    solana_clock::Epoch,
    solana_pubkey::Pubkey,
    std::{collections::HashMap, convert::identity, ops::Index, sync::Arc},
};

mod identity_keyed;
mod vote_keyed;
pub use {
    identity_keyed::LeaderSchedule as IdentityKeyedLeaderSchedule,
    vote_keyed::LeaderSchedule as VoteKeyedLeaderSchedule,
};

// Used for testing
#[derive(Clone, Debug)]
pub struct FixedSchedule {
    pub leader_schedule: Arc<LeaderSchedule>,
}

/// Stake-weighted leader schedule for one epoch.
pub type LeaderSchedule = Box<dyn LeaderScheduleVariant>;

pub trait LeaderScheduleVariant:
    std::fmt::Debug + Send + Sync + Index<u64, Output = Pubkey>
{
    fn get_slot_leaders(&self) -> &[Pubkey];
    fn get_leader_slots_map(&self) -> &HashMap<Pubkey, Vec<usize>>;

    /// Get the vote account address for the given epoch slot index. This is
    /// guaranteed to be Some if the leader schedule is keyed by vote account
    fn get_vote_key_at_slot_index(&self, _epoch_slot_index: usize) -> Option<&Pubkey> {
        None
    }

    fn get_leader_upcoming_slots(
        &self,
        pubkey: &Pubkey,
        offset: usize, // Starting index.
    ) -> Box<dyn Iterator<Item = usize> + '_> {
        let index = self.get_leader_slots_map().get(pubkey);
        let num_slots = self.num_slots();

        match index {
            Some(index) if !index.is_empty() => {
                let size = index.len();
                let start_offset = index
                    .binary_search(&(offset % num_slots))
                    .unwrap_or_else(identity)
                    + offset / num_slots * size;
                // The modular arithmetic here and above replicate Index implementation
                // for LeaderSchedule, where the schedule keeps repeating endlessly.
                // The '%' returns where in a cycle we are and the '/' returns how many
                // times the schedule is repeated.
                Box::new(
                    (start_offset..=usize::MAX)
                        .map(move |k| index[k % size] + k / size * num_slots),
                )
            }
            _ => {
                // Empty iterator for pubkeys not in schedule
                #[allow(clippy::reversed_empty_ranges)]
                Box::new((1..=0).map(|_| 0))
            }
        }
    }

    fn num_slots(&self) -> usize {
        self.get_slot_leaders().len()
    }
}

// Note: passing in zero keyed stakes will cause a panic.
fn stake_weighted_slot_leaders(
    mut keyed_stakes: Vec<(&Pubkey, u64)>,
    epoch: Epoch,
    len: u64,
    repeat: u64,
) -> Vec<Pubkey> {
    debug_assert!(
        len.is_multiple_of(repeat),
        "expected `len` {len} to be divisible by `repeat` {repeat}"
    );
    sort_stakes(&mut keyed_stakes);
    let (keys, stakes): (Vec<_>, Vec<_>) = keyed_stakes.into_iter().unzip();
    let weighted_index = WeightedU64Index::new(stakes).unwrap();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&epoch.to_le_bytes());
    let rng = &mut ChaChaRng::from_seed(seed);
    let mut current_slot_leader = Pubkey::default();
    (0..len)
        .map(|i| {
            if i % repeat == 0 {
                current_slot_leader = keys[weighted_index.sample(rng)];
            }
            current_slot_leader
        })
        .collect()
}

fn sort_stakes(stakes: &mut Vec<(&Pubkey, u64)>) {
    // Sort first by stake. If stakes are the same, sort by pubkey to ensure a
    // deterministic result.
    // Note: Use unstable sort, because we dedup right after to remove the equal elements.
    stakes.sort_unstable_by(|(l_pubkey, l_stake), (r_pubkey, r_stake)| {
        if r_stake == l_stake {
            r_pubkey.cmp(l_pubkey)
        } else {
            r_stake.cmp(l_stake)
        }
    });

    // Now that it's sorted, we can do an O(n) dedup.
    stakes.dedup();
}

/// Hybrid leader schedule generation algorithm that reduces variance in slot allocation.
///
/// This algorithm implements a two-phase allocation:
/// 1. **Deterministic floor**: Each validator receives floor(expected_rotations) rotations
///    based on their stake proportion.
/// 2. **Fractional lottery**: Remaining rotations are distributed via weighted random
///    selection based on each validator's fractional remainder.
///
/// Benefits:
/// - Maximum deviation per epoch is ±1 rotation (±4 slots with default repeat=4)
/// - Eliminates small validator starvation when stake justifies at least 1 rotation
/// - Fully deterministic and locally verifiable (same ChaChaRng seed as current algorithm)
///
/// # Arguments
/// * `keyed_stakes` - Vector of (pubkey, stake) pairs. Panics if all stakes are zero.
/// * `epoch` - The epoch number, used to seed the RNG for deterministic results.
/// * `len` - Total number of slots in the epoch schedule.
/// * `repeat` - Number of consecutive slots per leader (typically 4).
///
/// # Returns
/// A vector of pubkeys representing the leader for each slot.
///
/// # Example
/// For a validator with 3.75% stake in an epoch with 250 rotations:
/// - expected_rotations = 9.375
/// - base (floor) = 9 rotations guaranteed
/// - fraction = 0.375 (37.5% chance of winning 1 extra rotation in lottery)
/// - Final allocation: 9 or 10 rotations (36 or 40 slots)
pub fn hybrid_stake_weighted_slot_leaders(
    mut keyed_stakes: Vec<(&Pubkey, u64)>,
    epoch: Epoch,
    len: u64,
    repeat: u64,
) -> Vec<Pubkey> {
    debug_assert!(
        len.is_multiple_of(repeat),
        "expected `len` {len} to be divisible by `repeat` {repeat}"
    );

    sort_stakes(&mut keyed_stakes);

    let total_rotations = len / repeat;
    let total_stake: u64 = keyed_stakes.iter().map(|(_, stake)| stake).sum();

    assert!(total_stake > 0, "total stake must be greater than zero");

    // Phase 1: Calculate deterministic floor allocations and fractional remainders
    let mut allocations: Vec<(&Pubkey, u64)> = Vec::with_capacity(keyed_stakes.len());
    let mut fractions: Vec<(&Pubkey, f64)> = Vec::with_capacity(keyed_stakes.len());
    let mut rotations_allocated: u64 = 0;

    for (pubkey, stake) in &keyed_stakes {
        let expected_rotations =
            (*stake as f64 / total_stake as f64) * total_rotations as f64;
        let base = expected_rotations.floor() as u64;
        let fraction = expected_rotations - base as f64;

        allocations.push((pubkey, base));
        rotations_allocated += base;

        // Only include in lottery if there's a meaningful fractional part
        if fraction > 0.0 {
            fractions.push((pubkey, fraction));
        }
    }

    // Phase 2: Distribute remaining rotations via weighted lottery
    // Each validator can win at most ONE extra rotation to maintain the ±1 rotation bound
    let mut rotations_left = total_rotations.saturating_sub(rotations_allocated);

    if rotations_left > 0 && !fractions.is_empty() {
        // Create RNG with same seeding as original algorithm
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&epoch.to_le_bytes());
        let rng = &mut ChaChaRng::from_seed(seed);

        // Create a map for quick lookup of allocation index by pubkey
        let pubkey_to_index: HashMap<&Pubkey, usize> = allocations
            .iter()
            .enumerate()
            .map(|(i, (pk, _))| (*pk, i))
            .collect();

        // Track which validators have already won (each can win at most once)
        let mut remaining_fractions = fractions.clone();

        while rotations_left > 0 && !remaining_fractions.is_empty() {
            // Sample winner using weighted lottery based on fractional remainders
            let winner_idx = weighted_fraction_sample_index(&remaining_fractions, rng);
            let winner_pubkey = remaining_fractions[winner_idx].0;

            // Increment winner's allocation
            if let Some(&idx) = pubkey_to_index.get(winner_pubkey) {
                allocations[idx].1 += 1;
            }

            // Remove winner from the pool - they can only win once
            remaining_fractions.swap_remove(winner_idx);

            rotations_left -= 1;
        }
    }

    // Phase 3: Build the slot schedule from allocations
    // First, create a list of (pubkey, rotation_count) and shuffle the rotations
    // to distribute leaders throughout the epoch rather than having them clustered
    let mut rotation_assignments: Vec<&Pubkey> = Vec::with_capacity(total_rotations as usize);
    for (pubkey, count) in &allocations {
        for _ in 0..*count {
            rotation_assignments.push(pubkey);
        }
    }

    // Shuffle the rotation assignments deterministically using the same RNG seed
    // but with a different offset to avoid correlation with lottery draws
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&epoch.to_le_bytes());
    seed[8..16].copy_from_slice(&1u64.to_le_bytes()); // Different seed offset for shuffle
    let rng = &mut ChaChaRng::from_seed(seed);
    fisher_yates_shuffle(&mut rotation_assignments, rng);

    // Expand rotations into individual slots
    rotation_assignments
        .into_iter()
        .flat_map(|pubkey| std::iter::repeat_n(*pubkey, repeat as usize))
        .collect()
}

/// Perform weighted random selection based on fractional weights.
/// Returns the index of the winning entry in the fractions slice.
fn weighted_fraction_sample_index(
    fractions: &[(&Pubkey, f64)],
    rng: &mut ChaChaRng,
) -> usize {
    let total_weight: f64 = fractions.iter().map(|(_, f)| f).sum();

    // Generate a random value in [0, total_weight)
    let random_value: f64 = rng.random::<f64>() * total_weight;

    // Find which validator's range the random value falls into
    let mut cumulative = 0.0;
    for (i, (_, fraction)) in fractions.iter().enumerate() {
        cumulative += fraction;
        if random_value < cumulative {
            return i;
        }
    }

    // Fallback to last validator (should only happen due to floating point edge cases)
    fractions.len().saturating_sub(1)
}

/// Fisher-Yates shuffle for deterministic shuffling.
fn fisher_yates_shuffle<T>(slice: &mut [T], rng: &mut ChaChaRng) {
    for i in (1..slice.len()).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use {super::*, itertools::Itertools, rand::Rng, std::iter::repeat_with, test_case::test_case};

    #[test]
    fn test_get_leader_upcoming_slots() {
        const NUM_SLOTS: usize = 97;
        let mut rng = rand::rng();
        let pubkeys: Vec<_> = repeat_with(Pubkey::new_unique).take(4).collect();
        let schedule: Vec<_> = repeat_with(|| pubkeys[rng.random_range(0..3)])
            .take(19)
            .collect();
        let schedule = IdentityKeyedLeaderSchedule::new_from_schedule(schedule);
        let leaders = (0..NUM_SLOTS)
            .map(|i| (schedule[i as u64], i))
            .into_group_map();
        for pubkey in &pubkeys {
            let index = leaders.get(pubkey).cloned().unwrap_or_default();
            for offset in 0..NUM_SLOTS {
                let schedule: Vec<_> = schedule
                    .get_leader_upcoming_slots(pubkey, offset)
                    .take_while(|s| *s < NUM_SLOTS)
                    .collect();
                let index: Vec<_> = index.iter().copied().skip_while(|s| *s < offset).collect();
                assert_eq!(schedule, index);
            }
        }
    }

    #[test]
    fn test_sort_stakes_basic() {
        let pubkey0 = solana_pubkey::new_rand();
        let pubkey1 = solana_pubkey::new_rand();
        let mut stakes = vec![(&pubkey0, 1), (&pubkey1, 2)];
        sort_stakes(&mut stakes);
        assert_eq!(stakes, vec![(&pubkey1, 2), (&pubkey0, 1)]);
    }

    #[test]
    fn test_sort_stakes_with_dup() {
        let pubkey0 = solana_pubkey::new_rand();
        let pubkey1 = solana_pubkey::new_rand();
        let mut stakes = vec![(&pubkey0, 1), (&pubkey1, 2), (&pubkey0, 1)];
        sort_stakes(&mut stakes);
        assert_eq!(stakes, vec![(&pubkey1, 2), (&pubkey0, 1)]);
    }

    #[test]
    fn test_sort_stakes_with_equal_stakes() {
        let pubkey0 = Pubkey::default();
        let pubkey1 = solana_pubkey::new_rand();
        let mut stakes = vec![(&pubkey0, 1), (&pubkey1, 1)];
        sort_stakes(&mut stakes);
        assert_eq!(stakes, vec![(&pubkey1, 1), (&pubkey0, 1)]);
    }

    fn pubkey_from_u16(n: u16) -> Pubkey {
        let mut bytes = [0; 32];
        bytes[0..2].copy_from_slice(&n.to_le_bytes());
        Pubkey::new_from_array(bytes)
    }

    #[test_case(1, &[10, 20, 30], 12, 1, &[1, 1, 2, 1, 1, 0, 0, 1, 2, 1, 0, 1])]
    #[test_case(1, &[10, 20, 30], 12, 2, &[1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0])]
    #[test_case(1, &[30, 10, 20], 12, 1, &[2, 2, 0, 2, 2, 1, 1, 2, 0, 2, 1, 2])]
    #[test_case(1, &[30, 10, 20], 12, 2, &[2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 1,1])]
    #[test_case(1, &[10, 20, 25, 30], 12, 1, &[2, 2, 3, 1, 2, 0, 1, 1, 3, 2, 1, 2])]
    #[test_case(1, &[10, 20, 25, 30, 35, 40, 100], 15, 1,
                &[4, 5, 6, 3, 4, 1, 2, 3, 6, 4, 2, 4, 5, 6, 6])]
    #[test_case(1, &[10, 20, 25, 30, 35, 40, 100, 1000], 15, 1,
                &[7, 7, 7, 7, 7, 4, 6, 7, 7, 7, 6, 7, 7, 7, 7])]
    #[test_case(1, &[10, 20, 25, 30, 35, 40, 100, 1000, 10_000], 20, 1,
                &[8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7])]
    #[test_case(1, &[10, 20, 25, 30, 35, 40, 100, 1000, 10_000], 25, 1,
                &[8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8])]
    #[test_case(457468, &[10, 20, 30], 12, 1, &[2, 2, 0, 1, 0, 2, 1, 2, 1, 2, 2, 2])]
    #[test_case(457468, &[10, 20, 30], 12, 2, &[2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2])]
    #[test_case(457469, &[10, 20, 30], 12, 1, &[1, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0])]
    #[test_case(457470, &[10, 20, 30], 12, 1, &[2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2])]
    #[test_case(3466545, &[10, 20, 30], 12, 1, &[2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2])]
    #[test_case(3466545, &[10, 20, 30], 13, 1, &[2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1])]
    #[test_case(3466545, &[10, 20, 30], 14, 1, &[2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1, 2])]
    #[test_case(3466545, &[10, 20, 30], 14, 2, &[2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1])]
    fn test_stake_leader_schedule_exact_order(
        epoch: u64,
        stakes: &[u64],
        len: u64,
        repeat: u64,
        expected_order: &[usize],
    ) {
        let pubkeys: Vec<_> = (0..stakes.len() as u16).map(pubkey_from_u16).collect();
        let stakes = pubkeys.iter().zip(stakes.iter().copied()).collect();
        let order: Vec<_> = stake_weighted_slot_leaders(stakes, epoch, len, repeat)
            .into_iter()
            .map(|pubkey| {
                pubkeys
                    .iter()
                    .find_position(|item| *item == &pubkey)
                    .unwrap()
                    .0
            })
            .collect();
        assert_eq!(order, expected_order);
    }

    #[test_case(42, 1_000, 0, "4XU6LEarBUmBkAvXRsjeyLu3N8CcgrvbRFrNiJi2jECk")]
    #[test_case(42, 10_000, 0, "G2MGFXgdLATXWr1336i8PTcaUMc4GbJRMJdbxiarCttr")]
    #[test_case(42, 10_000, 1, "9xLLKyyqF5YrdwPSDbqh5oVamSF7cqPqQLxEyHTexEiP")]
    #[test_case(42, 10_000, 2, "AJ6NQi2p5SnRz9mqESqkW2PwVoT2vYy1fmKdaHxNFUAf")]
    #[test_case(42, 10_000, 3, "2oLjZggMwDTQhzdB4KN5VQisyeRw6MZbBBdjosNZK5xR")]
    #[test_case(346436, 1_000, 0, "59SnXMS4NzTSib8TNykiJgFQBeAVxUqsAvQm7JtkodPQ")]
    #[test_case(346436, 1_000, 1, "BEB2nC9MBALPbgwGKGfHu6V88QG7doScx65cAd6VjnRk")]
    #[test_case(346436, 1_000, 2, "3aLE5S6xLEU9yg5EZQH27qrC86aC2dG8KLh4NbcapXpy")]
    #[test_case(346436, 1_000, 3, "H2bw3Y2AjxJyK7smy1ZBB4LJ7MY3i9bPQM3YdAChAww2")]
    #[test_case(454357, 10_000, 0, "4BLanrC5t7vzNXx62javKtjCmCkd8yZfZpVrjT4eUpNQ")]
    #[test_case(454357, 10_000, 1, "FyvbdxpVchendERMnzH2KDceqydpXtJarrfFXoLQEXgQ")]
    #[test_case(454357, 10_000, 2, "7KwK44Y7V3GzJLN8aGZtM8EEfAYmRvaiDyKYV6jg4MQn")]
    #[test_case(454357, 10_000, 3, "E9XL5BLhCJ4Emyfs8jTUsQetfA8QZj78LcnN63dPp7jJ")]
    fn test_long_leader_schedule_hashed(
        epoch: Epoch,
        len: u64,
        stake_pow: u32,
        expected_hash: &str,
    ) {
        fn hash_pubkeys(v: &[Pubkey]) -> String {
            use sha2::{Digest, Sha256};

            let hasher = v.iter().fold(Sha256::new(), |hasher, pk| {
                hasher.chain_update(pk.to_bytes())
            });
            bs58::encode(hasher.finalize()).into_string()
        }
        let pubkeys: Vec<_> = (0..=u16::MAX).map(pubkey_from_u16).collect();
        let stakes = pubkeys
            .iter()
            .enumerate()
            .map(|(i, pk)| (pk, i.pow(stake_pow) as u64))
            .collect();
        let schedule = stake_weighted_slot_leaders(stakes, epoch, len, 1);
        assert_eq!(hash_pubkeys(&schedule), expected_hash);
    }

    #[test]
    #[should_panic]
    fn test_zero_stake_panics() {
        let _ = stake_weighted_slot_leaders(
            vec![(&pubkey_from_u16(1), 0), (&pubkey_from_u16(2), 0)],
            0,
            5,
            1,
        );
    }

    // ==================== Hybrid Algorithm Tests ====================

    #[test]
    fn test_hybrid_basic_allocation() {
        // Test with simple stakes that produce clean divisions
        let pubkeys: Vec<_> = (0..3u16).map(pubkey_from_u16).collect();
        // Stakes: 25%, 25%, 50% - with 100 rotations should give 25, 25, 50
        let stakes: Vec<_> = pubkeys.iter().zip([250u64, 250, 500].iter().copied()).collect();

        let schedule = hybrid_stake_weighted_slot_leaders(stakes, 0, 400, 4); // 100 rotations
        assert_eq!(schedule.len(), 400);

        // Count rotations per validator
        let mut counts: HashMap<Pubkey, u64> = HashMap::new();
        for pubkey in &schedule {
            *counts.entry(*pubkey).or_default() += 1;
        }

        // Each validator should get their floor allocation ± 1 rotation (4 slots)
        // pk0: 25% of 100 = 25 rotations = 100 slots
        // pk1: 25% of 100 = 25 rotations = 100 slots
        // pk2: 50% of 100 = 50 rotations = 200 slots
        assert_eq!(*counts.get(&pubkeys[0]).unwrap_or(&0), 100);
        assert_eq!(*counts.get(&pubkeys[1]).unwrap_or(&0), 100);
        assert_eq!(*counts.get(&pubkeys[2]).unwrap_or(&0), 200);
    }

    #[test]
    fn test_hybrid_max_deviation_is_one_rotation() {
        // Test that no validator deviates by more than ±1 rotation (±4 slots)
        let num_validators = 100;
        let pubkeys: Vec<_> = (0..num_validators as u16).map(pubkey_from_u16).collect();

        // Create varied stakes
        let stakes: Vec<_> = pubkeys
            .iter()
            .enumerate()
            .map(|(i, pk)| (pk, ((i + 1) * 1000) as u64))
            .collect();

        let total_stake: u64 = stakes.iter().map(|(_, s)| s).sum();
        let total_slots = 432_000u64; // Typical epoch length
        let repeat = 4u64;
        let total_rotations = total_slots / repeat;

        for epoch in [0u64, 100, 12345, 999999] {
            let schedule = hybrid_stake_weighted_slot_leaders(stakes.clone(), epoch, total_slots, repeat);

            // Count slots per validator
            let mut slot_counts: HashMap<Pubkey, u64> = HashMap::new();
            for pubkey in &schedule {
                *slot_counts.entry(*pubkey).or_default() += 1;
            }

            // Check each validator's deviation
            for (pk, stake) in &stakes {
                let expected_rotations = (*stake as f64 / total_stake as f64) * total_rotations as f64;
                let expected_slots = expected_rotations * repeat as f64;
                let actual_slots = *slot_counts.get(*pk).unwrap_or(&0) as f64;

                let deviation_slots = (actual_slots - expected_slots).abs();
                let max_allowed_deviation = repeat as f64; // ±1 rotation = ±4 slots

                assert!(
                    deviation_slots <= max_allowed_deviation,
                    "Validator {:?} deviated by {} slots (max allowed: {}). Expected: {}, Got: {}",
                    pk,
                    deviation_slots,
                    max_allowed_deviation,
                    expected_slots,
                    actual_slots
                );
            }
        }
    }

    #[test]
    fn test_hybrid_deterministic() {
        // Same inputs should produce same output
        let pubkeys: Vec<_> = (0..10u16).map(pubkey_from_u16).collect();
        let stakes: Vec<_> = pubkeys.iter().zip((1..=10).map(|i| i * 100u64)).collect();

        let schedule1 = hybrid_stake_weighted_slot_leaders(stakes.clone(), 12345, 1000, 4);
        let schedule2 = hybrid_stake_weighted_slot_leaders(stakes, 12345, 1000, 4);

        assert_eq!(schedule1, schedule2);
    }

    #[test]
    fn test_hybrid_different_epochs_different_schedules() {
        // Different epochs should produce different schedules (different shuffle)
        let pubkeys: Vec<_> = (0..10u16).map(pubkey_from_u16).collect();
        let stakes: Vec<_> = pubkeys.iter().zip((1..=10).map(|i| i * 100u64)).collect();

        let schedule1 = hybrid_stake_weighted_slot_leaders(stakes.clone(), 100, 1000, 4);
        let schedule2 = hybrid_stake_weighted_slot_leaders(stakes, 101, 1000, 4);

        // Schedules should have same total counts but different ordering
        assert_ne!(schedule1, schedule2);
    }

    #[test]
    fn test_hybrid_respects_repeat() {
        // Verify that leaders repeat for `repeat` consecutive slots
        let pubkeys: Vec<_> = (0..5u16).map(pubkey_from_u16).collect();
        let stakes: Vec<_> = pubkeys.iter().zip([100u64, 200, 300, 400, 500]).collect();

        let repeat = 4u64;
        let schedule = hybrid_stake_weighted_slot_leaders(stakes, 0, 100, repeat);

        // Check that every group of `repeat` slots has the same leader
        for chunk in schedule.chunks(repeat as usize) {
            let first = &chunk[0];
            for slot_leader in chunk {
                assert_eq!(slot_leader, first, "Leader should repeat for {} consecutive slots", repeat);
            }
        }
    }

    #[test]
    fn test_hybrid_total_slots_correct() {
        let pubkeys: Vec<_> = (0..3u16).map(pubkey_from_u16).collect();
        let stakes: Vec<_> = pubkeys.iter().zip([100u64, 200, 300]).collect();

        let len = 432_000u64;
        let schedule = hybrid_stake_weighted_slot_leaders(stakes, 0, len, 4);

        assert_eq!(schedule.len() as u64, len);
    }

    #[test]
    fn test_hybrid_small_validator_gets_slots() {
        // Even a small validator should get at least their floor allocation
        let pubkeys: Vec<_> = (0..3u16).map(pubkey_from_u16).collect();
        // Small validator has 1% stake (100 out of 10000), large validators have 49.5% each
        let stakes: Vec<_> = pubkeys.iter().zip([100u64, 4950, 4950]).collect();

        let total_slots = 40_000u64;
        let repeat = 4u64;
        // total_rotations = 10,000

        // Small validator: 100/10000 = 1% of 10,000 rotations = 100 rotations expected
        let schedule = hybrid_stake_weighted_slot_leaders(stakes, 0, total_slots, repeat);

        let small_validator_slots = schedule.iter().filter(|pk| **pk == pubkeys[0]).count() as u64;
        let small_validator_rotations = small_validator_slots / repeat;

        // Should get floor(100) = 100 rotations, possibly + 1 from lottery
        // So between 100 and 101 rotations (400-404 slots)
        assert!(
            small_validator_rotations >= 100 && small_validator_rotations <= 101,
            "Small validator got {} rotations, expected ~100",
            small_validator_rotations
        );
    }

    #[test]
    #[should_panic(expected = "total stake must be greater than zero")]
    fn test_hybrid_zero_stake_panics() {
        let _ = hybrid_stake_weighted_slot_leaders(
            vec![(&pubkey_from_u16(1), 0), (&pubkey_from_u16(2), 0)],
            0,
            100,
            4,
        );
    }

    #[test]
    fn test_hybrid_single_validator() {
        // Single validator should get all slots
        let pubkey = pubkey_from_u16(42);
        let stakes = vec![(&pubkey, 1000u64)];

        let schedule = hybrid_stake_weighted_slot_leaders(stakes, 0, 100, 4);

        assert!(schedule.iter().all(|pk| *pk == pubkey));
        assert_eq!(schedule.len(), 100);
    }

    #[test]
    fn test_hybrid_variance_comparison() {
        // Compare variance between original and hybrid algorithms
        // This is more of a demonstration/documentation test

        let num_validators = 50;
        let pubkeys: Vec<_> = (0..num_validators as u16).map(pubkey_from_u16).collect();

        // Create realistic stake distribution (power law-ish)
        let stakes: Vec<_> = pubkeys
            .iter()
            .enumerate()
            .map(|(i, pk)| (pk, ((i + 1).pow(2) * 100) as u64))
            .collect();

        let total_stake: u64 = stakes.iter().map(|(_, s)| s).sum();
        let total_slots = 432_000u64;
        let repeat = 4u64;

        // Run hybrid algorithm
        let hybrid_schedule = hybrid_stake_weighted_slot_leaders(stakes.clone(), 12345, total_slots, repeat);

        // Count slots per validator in hybrid
        let mut hybrid_counts: HashMap<Pubkey, u64> = HashMap::new();
        for pubkey in &hybrid_schedule {
            *hybrid_counts.entry(*pubkey).or_default() += 1;
        }

        // Calculate max deviation for hybrid
        let mut max_hybrid_deviation: f64 = 0.0;
        for (pk, stake) in &stakes {
            let expected = (*stake as f64 / total_stake as f64) * total_slots as f64;
            let actual = *hybrid_counts.get(*pk).unwrap_or(&0) as f64;
            let deviation = (actual - expected).abs();
            max_hybrid_deviation = max_hybrid_deviation.max(deviation);
        }

        // Hybrid max deviation should be at most 1 rotation (4 slots)
        assert!(
            max_hybrid_deviation <= repeat as f64,
            "Hybrid max deviation {} exceeds {} slots",
            max_hybrid_deviation,
            repeat
        );
    }
}