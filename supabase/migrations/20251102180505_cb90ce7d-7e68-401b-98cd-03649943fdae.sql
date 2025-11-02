-- Allow anyone to read rl_q_state for dashboard display
CREATE POLICY "Anyone can read rl_q_state"
ON public.rl_q_state
FOR SELECT
TO public
USING (true);