export function useFusedBeliefPropagate(): boolean {
  return globalThis.process?.env?.P2_DISABLE_FUSED_BELIEF_PROPAGATE !== "1";
}

export function useAllInOverlapList(): boolean {
  return globalThis.process?.env?.P2_DISABLE_ALLIN_OVERLAP_LIST !== "1";
}

export function useAllInNoOpponentAllowed(): boolean {
  return globalThis.process?.env?.P2_DISABLE_ALLIN_NO_OPP_ALLOWED !== "1";
}
