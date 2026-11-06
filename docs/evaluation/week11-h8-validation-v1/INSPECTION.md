# What the saved adapter gains and regressions show

All nine recordings for clean107, clean151 and targeted231 were inspected using
saved cameras and reconstructed physical states. No new policy calls, rendered
camera observations or simulator integration were added. The maximum saved-state
round-trip error was 2.22e-16. All nine condition-pair prefixes were already
matched in the stopped-screen audit.

## Clean107: both adapters improve retention during lifting

The task is to open the top drawer and put the bowl inside. All three policies
open that drawer. Fixed H8 lifts the bowl only 103 mm, with bilateral finger
contact at frames 145–158, then loses it while still commanding close. Its first
opening command occurs at frame 183, after the loss. This is not an early
commanded-release failure.

The horizon-aware adapter lifts the bowl 289 mm and retains bilateral contact
through frame 191, apart from a one-frame interruption. The H8-information
control lifts it 294 mm with bilateral contact through frame 189. Both complete
the task. The matched control therefore reproduces this gain without receiving
non-8 future tokens; extra learned motion capacity is a plausible contributor.
The comparison does not establish that future-horizon information is necessary.

[Matched saved cameras](inspection-mirror/inspection-focus/clean-200-0107-matched-window.png)
show fixed H8 losing the bowl between frames 158 and 162 while both adapters
continue carrying it toward the open drawer.

## Clean151: both adapters lose a late successful grasp recovery

All policies turn on the stove, first satisfying the reconstructed stove-on
predicate at frames 99, 101 and 100 for fixed H8, main adapter and control.
The difference occurs during pot acquisition afterward.

Fixed H8 eventually obtains bilateral pot contact at frame 298, raises the pot
105 mm, and succeeds. Neither adapter ever obtains bilateral pot contact in
its 530-frame recording. Their maximum pot height changes are only 6.4 and
4.8 mm. [Matched cameras](inspection-mirror/inspection-focus/clean-200-0151-matched-window.png)
show fixed H8's later recovery and transport while both adapters continue
unsuccessful grasp attempts. The regression is shared by both adapters, so it
does not isolate non-8 information as the cause.

Freezing gripper weights does not preserve the gripper trajectory: changed
motion changes the observation, and the frozen gripper rule can then issue
different commands. Identical-input H8 preservation remains satisfied, but it
cannot guarantee recovery after an earlier adapter intervention changes state.

## Targeted231: the main adapter pulls the wrong drawer

The instruction requests the middle drawer. The H8-information control pulls
that drawer by about 138 mm in the last saved pre-action state and completes
the task on the following action. Fixed H8 never appreciably opens it.

The main adapter instead pulls the bottom drawer about 161 mm; the middle
drawer remains near its closed position. This is visible in the
[matched camera sheet](inspection-mirror/inspection-focus/targeted-467-0231-matched-window.png)
and independently present in the reconstructed drawer joints. The two adapters
have identical executed gripper commands through their shared 181-frame span.
Their first differing motion action is at frame 26 on matching inputs and the
same selected horizon, 32. Different spatial control, rather than a difference
in the initial event forecast, is the observed distinction.

This shows a useful capacity control outperforming the main adapter on the
requested object/fixture. Because their trained weights differ as well as their
future-token inputs, it is not an isolated causal ablation of token information.

## Implication

The observed improvement is selective: better bowl retention coexists with
lost pot-grasp recovery and a wrong-drawer intervention. These findings do not
support a universal gripper rule, a case-specific time gate, or more training
without a preservation hypothesis. The event-free fixed-H8 validation is the
more direct test of useful performance at vanilla-like cost. If adapters are
revisited, a controlled study of preserving reference motion during correction
training is more motivated than simply increasing the update count.

Frames are zero-based pre-action states, including ten settling frames; intervals
above use inclusive endpoints for readability. The terminal post-action state
is not stored, so reconstructed goal flags can remain false on a successful
episode's final saved frame. Recorded terminal outcomes remain authoritative.
Reconstruction lacks the original model XML; these are automated physical
measurements plus assisted visual review, not independent human ground-truth labels.

[Detailed measurements](inspection-mirror/inspection/inspection.json),
[annotation completion](inspection-mirror/annotations/completion.json), and
[overview/trace figures](inspection-mirror/inspection/completion.json) preserve
the evidence. The chosen camera windows are post-hoc diagnostic selections.
