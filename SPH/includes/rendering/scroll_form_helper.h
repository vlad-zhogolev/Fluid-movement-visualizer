#pragma once

#include <nanogui/nanogui.h>

NAMESPACE_BEGIN(nanogui)

class ScrollFormHelper
{
public:
    /// Create a helper class to construct NanoGUI widgets on the given screen
    ScrollFormHelper(Screen *screen) : mScreen(screen)
    {
    }

    /// Add a new top-level window
    Window *addWindow(const Vector2i &pos,
        const std::string &title = "Untitled")
    {
        assert(mScreen);
        mWindow = new Window(mScreen, title);

        mScroll = new nanogui::VScrollPanel(mWindow);
        //mScroll->setFixedSize({ 200, 300 });
        mScroll->setFixedHeight(330);
        // mScroll->setWidth(200);
        mScroll->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 2, 8));
        mWrapper = new nanogui::Widget(mScroll);

        mLayout = new AdvancedGridLayout({ 10, 0, 10, 0 }, {});
        mLayout->setMargin(10);
        mLayout->setColStretch(2, 1);
        mWrapper->setLayout(mLayout);

        mWindow->setPosition(pos);
        mWindow->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 2, 8));
        //mWindow->setFixedSize({ 200, 200 });
        mWindow->setFixedHeight(370);
        mWindow->setVisible(true);
        return mWindow;
    }

    /// Add a new group that may contain several sub-widgets
    Label *addGroup(const std::string &caption)
    {
        Label* label = new Label(mWrapper, caption, mGroupFontName, mGroupFontSize);
        if (mLayout->rowCount() > 0)
            mLayout->appendRow(mPreGroupSpacing); /* Spacing */
        mLayout->appendRow(0);
        mLayout->setAnchor(label, AdvancedGridLayout::Anchor(0, mLayout->rowCount() - 1, 4, 1));
        mLayout->appendRow(mPostGroupSpacing);
        return label;
    }

    /// Add a new data widget controlled using custom getter/setter functions
    template <typename Type> detail::FormWidget<Type> *
        addVariable(const std::string &label, const std::function<void(const Type &)> &setter,
            const std::function<Type()> &getter, bool editable = true)
    {
        Label *labelW = new Label(mWrapper, label, mLabelFontName, mLabelFontSize);
        auto widget = new detail::FormWidget<Type>(mWrapper);
        auto refresh = [widget, getter] {
            Type value = getter(), current = widget->value();
            if (value != current)
                widget->setValue(value);
        };
        refresh();
        widget->setCallback(setter);
        widget->setEditable(editable);
        widget->setFontSize(mWidgetFontSize);
        Vector2i fs = widget->fixedSize();
        widget->setFixedSize(Vector2i(fs.x() != 0 ? fs.x() : mFixedSize.x(),
            fs.y() != 0 ? fs.y() : mFixedSize.y()));
        mRefreshCallbacks.push_back(refresh);
        if (mLayout->rowCount() > 0)
            mLayout->appendRow(mVariableSpacing);
        mLayout->appendRow(0);
        mLayout->setAnchor(labelW, AdvancedGridLayout::Anchor(1, mLayout->rowCount() - 1));
        mLayout->setAnchor(widget, AdvancedGridLayout::Anchor(3, mLayout->rowCount() - 1));
        return widget;
    }

    /// Add a new data widget that exposes a raw variable in memory
    template <typename Type> detail::FormWidget<Type> *
        addVariable(const std::string &label, Type &value, bool editable = true)
    {
        return addVariable<Type>(label,
            [&](const Type & v) { value = v; },
            [&]() -> Type { return value; },
            editable
            );
    }

    /// Add a button with a custom callback
    Button *addButton(const std::string &label, const std::function<void()> &cb)
    {
        Button *button = new Button(mWrapper, label);
        button->setCallback(cb);
        button->setFixedHeight(25);
        if (mLayout->rowCount() > 0)
            mLayout->appendRow(mVariableSpacing);
        mLayout->appendRow(0);
        mLayout->setAnchor(button, AdvancedGridLayout::Anchor(1, mLayout->rowCount() - 1, 3, 1));
        return button;
    }

    /// Add an arbitrary (optionally labeled) widget to the layout
    void addWidget(const std::string &label, Widget *widget)
    {
        mLayout->appendRow(0);
        if (label == "")
        {
            mLayout->setAnchor(widget, AdvancedGridLayout::Anchor(1, mLayout->rowCount() - 1, 3, 1));
        }
        else
        {
            Label *labelW = new Label(mWrapper, label, mLabelFontName, mLabelFontSize);
            mLayout->setAnchor(labelW, AdvancedGridLayout::Anchor(1, mLayout->rowCount() - 1));
            mLayout->setAnchor(widget, AdvancedGridLayout::Anchor(3, mLayout->rowCount() - 1));
        }
    }

    /// Cause all widgets to re-synchronize with the underlying variable state
    void refresh()
    {
        for (auto const &callback : mRefreshCallbacks)
            callback();
    }

    /// Access the currently active \ref Window instance
    Window *window()
    {
        return mWindow;
    }

    Widget* wrapper()
    {
        return mWrapper;
    }

    /// Set the active \ref Window instance.
    void setWindow(Window *window)
    {
        mWindow = window;
        mLayout = dynamic_cast<AdvancedGridLayout *>(window->layout());
        if (mLayout == nullptr)
            throw std::runtime_error(
                "Internal error: window has an incompatible layout!");
    }

    /// Specify a fixed size for newly added widgets.
    void setFixedSize(const Vector2i &fw)
    {
        mFixedSize = fw;
    }

    /// The current fixed size being used for newly added widgets.
    Vector2i fixedSize()
    {
        return mFixedSize;
    }

    /// The font name being used for group headers.
    const std::string &groupFontName() const
    {
        return mGroupFontName;
    }

    /// Sets the font name to be used for group headers.
    void setGroupFontName(const std::string &name)
    {
        mGroupFontName = name;
    }

    /// The font name being used for labels.
    const std::string &labelFontName() const
    {
        return mLabelFontName;
    }

    /// Sets the font name being used for labels.
    void setLabelFontName(const std::string &name)
    {
        mLabelFontName = name;
    }

    /// The size of the font being used for group headers.
    int groupFontSize() const
    {
        return mGroupFontSize;
    }

    /// Sets the size of the font being used for group headers.
    void setGroupFontSize(int value)
    {
        mGroupFontSize = value;
    }

    /// The size of the font being used for labels.
    int labelFontSize() const
    {
        return mLabelFontSize;
    }

    /// Sets the size of the font being used for labels.
    void setLabelFontSize(int value)
    {
        mLabelFontSize = value;
    }

    /// The size of the font being used for non-group / non-label widgets.
    int widgetFontSize() const
    {
        return mWidgetFontSize;
    }

    /// Sets the size of the font being used for non-group / non-label widgets.
    void setWidgetFontSize(int value)
    {
        mWidgetFontSize = value;
    }

protected:
    /// A reference to the \ref nanogui::Screen this FormHelper is assisting.
    ref<Screen> mScreen;

    /// A reference to the \ref nanogui::Window this FormHelper is controlling.
    ref<Window> mWindow;

    ref<VScrollPanel> mScroll;

    ///
    ref<Widget> mWrapper;

    /// A reference to the \ref nanogui::AdvancedGridLayout this FormHelper is using.
    ref<AdvancedGridLayout> mLayout;

    /// The callbacks associated with all widgets this FormHelper is managing.
    std::vector<std::function<void()>> mRefreshCallbacks;

    /// The group header font name.
    std::string mGroupFontName = "sans-bold";

    /// The label font name.
    std::string mLabelFontName = "sans";

    /// The fixed size for newly added widgets.
    Vector2i mFixedSize = Vector2i(0, 20);

    /// The font size for group headers.
    int mGroupFontSize = 20;

    /// The font size for labels.
    int mLabelFontSize = 16;

    /// The font size for non-group / non-label widgets.
    int mWidgetFontSize = 16;

    /// The spacing used **before** new groups.
    int mPreGroupSpacing = 15;

    /// The spacing used **after** each group.
    int mPostGroupSpacing = 5;

    /// The spacing between all other widgets.
    int mVariableSpacing = 5;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

NAMESPACE_END(nanogui)
