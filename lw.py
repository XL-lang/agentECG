I_twave = ecg.get_lead_segment('I', 'T wave')  # pyright: ignore[reportUndefinedVariable]
if (mean(I_twave) < 0.1):  # pyright: ignore[reportUndefinedVariable]
    print("T-wave amplitude is less than 0.1mv")
else:
    print("T-wave amplitude is greater than 0.1mv")