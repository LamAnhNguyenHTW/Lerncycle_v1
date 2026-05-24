export const BETA_NOT_INVITED_ERROR_DE =
  'Diese E-Mail ist nicht für die Beta freigegeben.';

export const BETA_NOT_INVITED_ERROR_EN =
  'This email is not on the Beta access list.';

type BetaInviteRpcClient = {
  rpc: (
    fn: 'is_beta_invited',
    args: {candidate_email: string},
  ) => PromiseLike<{data: boolean | null; error: {message: string} | null}>;
};

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export async function isEmailBetaInvited(
  supabase: BetaInviteRpcClient,
  email: string,
): Promise<boolean> {
  const {data, error} = await supabase.rpc('is_beta_invited', {
    candidate_email: normalizeEmail(email),
  });

  if (error) {
    throw new Error(error.message);
  }

  return data === true;
}
