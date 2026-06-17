/* Let's attempt to make micrograd in C++*/

class Value
{
    float myValue;
    float myGrad;
    Value* listOfParents; //here we're going to store the references to all of the values that were required to make this node into a list (potentially linked)
    int numParents;
    using myBackPropFunc = int(*)(Value*, Value*); //function pointer which will be determined after an operation has been performed


public:
    Value(float initialValue);
    ~Value();

    void setGrad(float newGrad);
};

Value::Value(float initialValue)
{
    myValue = initialValue;
    myGrad = 0.0;
    numParents = 0;
}

void Value::setGrad(float newGrad)
{
    myGrad = newGrad;
}




Value::~Value()
{
}

