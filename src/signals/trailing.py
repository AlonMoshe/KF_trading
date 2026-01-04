# src/signals/trailing.py

class DirectionalTrailing:
    """
    Generic directional trailing engine.

    Tracks a series and counts consecutive bars moving
    in a specified direction.

    Semantics:
    - side="long": favorable if value increases
    - side="short": favorable if value decreases

    Execution fires when the number of consecutive
    favorable or unfavorable bars (depending on mode)
    reaches n_consecutive.

    This class is intentionally generic to support
    multiple trailing "flavors" via parameters.
    """


    def __init__(
        self,
        *,
        level_col: str,
        n_consecutive: int,
        count_mode: str = "against",   # "against" | "with"
        reset_on_favorable: bool = True,
        activation_delay: int = 0,
        logger=None,
    ):

    

        if n_consecutive < 1:
            raise ValueError("n_consecutive must be >= 1")

        self.level_col = level_col
        self.n_consecutive = int(n_consecutive)
        self.count_mode = count_mode
        self.reset_on_favorable = reset_on_favorable
        self.activation_delay = activation_delay
        self.logger = logger

        self.reset()


    # --------------------------------------------------
    # State management
    # --------------------------------------------------
    def reset(self):
        self.active = False
        self.side = None
        self.counter = 0
        self.last_level = None
        self.execute_flag = False
        self.armed_index = None
        self.bars_since_arm = 0


    # --------------------------------------------------
    # Arm trailing after entry signal
    # --------------------------------------------------
    def arm(self, *, side: str, index: int, level_value: float, ts:str):
        """
        Start trailing after an entry signal.

        side: "long" or "short"
        index: bar index where arming occurs
        level_value: KF level at arming bar
        """
        

        
        self.reset()

        self.active = True
        self.side = side
        self.counter = 0
        self.last_level = level_value
        self.armed_index = index
        self.bars_since_arm = 0


        if self.logger is not None:
            self.logger.log(
                ts,
                f"TRAILING ARMED side={side} "
                f"level={level_value:.6f}"
            )

    # --------------------------------------------------
    # Update trailing state each bar
    # --------------------------------------------------
    def update(self, *, index: int, level_value: float, ts:str):
        if not self.active or self.execute_flag:
            return
        self.bars_since_arm += 1
        if self.bars_since_arm <= self.activation_delay:
            # still warming up; just update last_level and return
            self.last_level = level_value
            return

        
        if self.side == "long":
            favorable = level_value > self.last_level
        elif self.side == "short":
            favorable = level_value < self.last_level
        else:
            raise RuntimeError("Invalid trailing side")

        # count_mode determines what increments the counter
        if self.count_mode == "with":
            count_event = favorable
        elif self.count_mode == "against":
            count_event = not favorable
        else:
            raise RuntimeError("Invalid count_mode")

        if count_event:
            self.counter += 1
        else:
            self.counter = 0


        if self.logger is not None:
            self.logger.log(
                ts,
                f"TRAIL DEBUG side={self.side} "
                f"level={level_value:.6f} "
                f"last_level={self.last_level:.6f} "
                f"counter={self.counter}/{self.n_consecutive}"
            )

        self.last_level = level_value

        if self.counter >= self.n_consecutive:
            self.execute_flag = True

            if self.logger is not None:
                self.logger.log(
                    ts,
                    f"TRAIL EXECUTE side={self.side} "
                    f"counter={self.counter}"
                )

    # --------------------------------------------------
    # Query execution condition
    # --------------------------------------------------
    def should_execute(self) -> bool:
        return self.execute_flag
