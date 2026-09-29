# Gate A (H1 replication)

- **low**: ticket beats the best trained baseline at 4/4 sparsities (49.0%, 24.0%, 11.8%, 5.8%); expected >= 3 -> PASS
- **warm03**: ticket beats the best trained baseline at 2/2 sparsities (49.0%, 24.0%); expected >= 3 -> FAIL
- **high**: winning ticket (advantage > 0 and acc >= dense - 0.5 pp) at 0/3 sparsities (); expected none -> PASS
  (positive advantage without a winning ticket at 3/3: 49.0% +0.36 pp, ticket -0.62 pp vs dense, 24.0% +0.13 pp, ticket -2.55 pp vs dense, 11.8% +0.39 pp, ticket -4.31 pp vs dense)

**Gate A: NOT PASSED** (all verdicts are in ticket_advantage.csv)
